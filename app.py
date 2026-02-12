import io
import json
import os
import re
import shutil
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import math
import hashlib
from datetime import datetime
from bs4 import BeautifulSoup
from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# ============================================================
# Paths / App
# ============================================================
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
TMP_DIR = DATA_DIR / "tmp"

for d in (DATA_DIR, UPLOAD_DIR, SNAPSHOT_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

# ============================================================
# Seasons
# ============================================================
SEASONS: Dict[str, str] = {
    "V26 - ACT 1": "3ea2b318-423b-cf86-25da-7cbb0eefbe2d",
    "V25 - ACT 6": "4c4b8cff-43eb-13d3-8f14-96b783c90cd2",
    "V25 - ACT 5": "5adc33fa-4f30-2899-f131-6fba64c5dd3a",
}
DEFAULT_SEASON_ID = SEASONS["V26 - ACT 1"]

# ============================================================
# Agent roles (best-effort, used for comp heuristics)
# ============================================================
def _agent_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())

AGENT_ROLES = {
    "Astra": "Controller",
    "Brimstone": "Controller",
    "Clove": "Controller",
    "Harbor": "Controller",
    "Omen": "Controller",
    "Viper": "Controller",
    "Breach": "Initiator",
    "Fade": "Initiator",
    "Gekko": "Initiator",
    "KAY/O": "Initiator",
    "Skye": "Initiator",
    "Sova": "Initiator",
    "Chamber": "Sentinel",
    "Cypher": "Sentinel",
    "Deadlock": "Sentinel",
    "Killjoy": "Sentinel",
    "Sage": "Sentinel",
    "Iso": "Duelist",
    "Jett": "Duelist",
    "Neon": "Duelist",
    "Phoenix": "Duelist",
    "Raze": "Duelist",
    "Reyna": "Duelist",
    "Yoru": "Duelist",
}

AGENT_ROLE_BY_KEY = {_agent_key(k): v for k, v in AGENT_ROLES.items()}
ROLE_ORDER = ["Controller", "Initiator", "Sentinel", "Duelist", "Flex"]

# ============================================================
# Snapshot helpers
# ============================================================
def snapshot_path(team_key: str) -> Path:
    return SNAPSHOT_DIR / f"{team_key}.json"

def to_float(x, default=0.0):
    """
    Robust float parser for messy scraped values like:
    None, "None", "", "\u2014", "-", "27.1%", "1,234", "  12.3  "
    """
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s or s.lower() in {"none", "null", "nan", "\u2014", "-", "n/a"}:
            return default
        s = s.replace("%", "").replace(",", "")
        return float(s)
    except Exception:
        return default

def to_int(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, int):
            return int(x)
        if isinstance(x, float):
            return int(x)
        s = str(x).strip().replace(",", "")
        if not s or s.lower() in {"none", "null", "\u2014", "-", "n/a"}:
            return default
        return int(float(s))
    except Exception:
        return default

def load_snapshot(team_key: str) -> dict:
    p = snapshot_path(team_key)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_snapshot(team_key: str, snap: dict) -> None:
    snapshot_path(team_key).write_text(json.dumps(snap, indent=2), encoding="utf-8")


def list_snapshots() -> List[Dict[str, Any]]:
    out = []
    for p in SNAPSHOT_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append(
            {
                "team_key": data.get("team_key") or p.stem,
                "team_name": data.get("team_name") or p.stem,
                "players": data.get("players") or [],
                "players_count": len(data.get("players") or []),
                "updated_ts": data.get("updated_ts"),
            }
        )
    out.sort(key=lambda x: (x.get("updated_ts") or 0), reverse=True)
    return out

DASHBOARD_DIR = DATA_DIR / "dashboards"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

def _now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _safe_div(a, b):
    return a / b if b else 0.0

def _weighted_avg(items, key, wkey="games"):
    num = 0.0
    den = 0
    for it in items:
        w = int(it.get(wkey, 0))
        num += float(it.get(key, 0.0)) * w
        den += w
    return _safe_div(num, den)

def _stdev(values):
    if not values:
        return 0.0
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(var)

def agent_role(agent: str) -> str:
    key = _agent_key(agent)
    return AGENT_ROLE_BY_KEY.get(key, "Flex")

def map_confidence_label(games: int) -> str:
    if games >= 12:
        return "High"
    if games >= 6:
        return "Medium"
    return "Low"

def map_confidence_class(label: str) -> str:
    if label == "High":
        return "pill-good"
    if label == "Medium":
        return "pill-mid"
    return "pill-bad"

def role_mix_label(role_counts: Dict[str, int]) -> str:
    parts = []
    for role in ROLE_ORDER:
        n = int(role_counts.get(role, 0))
        if n > 0:
            parts.append(f"{n} {role}")
    return " / ".join(parts) if parts else "Unknown"

def build_agent_pools(season_block: dict, roster: list) -> Tuple[Dict[str, list], Dict[str, int]]:
    pools = {}
    role_counts = {}
    for rid in roster:
        entry = season_block.get(rid) or {}
        agents = []
        for a in (entry.get("agents") or []):
            agent = (a.get("agent") or a.get("name") or "").strip()
            if not agent:
                continue
            matches = to_int(a.get("matches", 0))
            win_rate = to_float(a.get("win_rate", 0.0))
            acs = to_float(a.get("acs", 0.0))
            dmg_round = to_float(a.get("dmg_round", 0.0))
            role = agent_role(agent)
            if matches <= 0:
                continue
            agents.append({
                "agent": agent,
                "matches": matches,
                "win_rate": win_rate,
                "acs": acs,
                "dmg_round": dmg_round,
                "role": role,
            })
            role_counts[role] = role_counts.get(role, 0) + matches
        agents.sort(key=lambda x: x["matches"], reverse=True)
        pools[rid] = agents
    return pools, role_counts

def build_map_games_by_player(season_block: dict, roster: list) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for rid in roster:
        entry = season_block.get(rid) or {}
        for m in (entry.get("maps") or []):
            map_name = (m.get("map") or "").strip()
            if not map_name:
                continue
            games = int(m.get("wins", 0)) + int(m.get("losses", 0))
            if games <= 0:
                continue
            by_player = out.setdefault(map_name, {})
            by_player[rid] = by_player.get(rid, 0) + games
    return out

def pick_agents_for_players(player_ids: list, agent_pools: Dict[str, list]) -> Tuple[List[dict], int]:
    used = set()
    picks = []
    conflicts = 0
    for rid in player_ids:
        pool = agent_pools.get(rid) or []
        pick = None
        conflict = False
        for a in pool:
            if a["agent"] not in used:
                pick = a
                break
        if not pick and pool:
            pick = pool[0]
            conflict = True
            conflicts += 1
        if pick:
            if pick["agent"] in used:
                conflict = True
            used.add(pick["agent"])
            picks.append({
                "agent": pick["agent"],
                "role": pick["role"],
                "player": rid,
                "matches": pick["matches"],
                "win_rate": pick["win_rate"],
                "conflict": conflict,
            })
        else:
            conflicts += 1
            picks.append({
                "agent": "Unknown",
                "role": "Flex",
                "player": rid,
                "matches": 0,
                "win_rate": 0.0,
                "conflict": True,
            })
    return picks, conflicts

def build_expected_comps(
    map_rows: list,
    map_games_by_player: Dict[str, Dict[str, int]],
    agent_pools: Dict[str, list],
    fallback_player_order: list,
) -> List[dict]:
    comps = []
    for r in map_rows:
        map_name = (r.get("map") or "").strip()
        if not map_name:
            continue
        games = int(r.get("games", 0))
        by_player = map_games_by_player.get(map_name, {})
        if by_player:
            player_ids = [pid for pid, _ in sorted(by_player.items(), key=lambda x: x[1], reverse=True)]
        else:
            player_ids = list(fallback_player_order)
        player_ids = player_ids[:5]
        picks, conflicts = pick_agents_for_players(player_ids, agent_pools)
        role_counts = {}
        for p in picks:
            role = p.get("role") or "Flex"
            role_counts[role] = role_counts.get(role, 0) + 1
        conf = map_confidence_label(games)
        comps.append({
            "map": map_name,
            "games": games,
            "agents": picks,
            "roles_label": role_mix_label(role_counts),
            "confidence": conf,
            "confidence_class": map_confidence_class(conf),
            "conflicts": conflicts,
        })
    comps.sort(key=lambda x: x.get("games", 0), reverse=True)
    return comps

def compute_map_strengths(map_rows: list, smoothing: int = 6) -> List[dict]:
    rows = []
    for r in map_rows:
        games = int(r.get("games", 0))
        wins = int(r.get("wins", 0))
        win_rate = float(r.get("win_rate_pct", 0.0))
        adj = (wins + smoothing * 0.5) / (games + smoothing) if games > 0 else 0.5
        conf = map_confidence_label(games)
        rows.append({
            "map": r.get("map"),
            "games": games,
            "wins": wins,
            "losses": int(r.get("losses", 0)),
            "win_rate_pct": win_rate,
            "adj_win_rate_pct": adj * 100.0,
            "confidence": conf,
            "confidence_class": map_confidence_class(conf),
        })
    return rows

def recommend_map_plan(map_rows: list, min_games: int = 3, smoothing: int = 6, limit: int = 3) -> dict:
    rows = compute_map_strengths(map_rows, smoothing=smoothing)
    viable = [r for r in rows if r["games"] >= min_games]
    if not viable:
        viable = rows
    picks = sorted(viable, key=lambda r: r["adj_win_rate_pct"], reverse=True)
    bans = sorted(viable, key=lambda r: r["adj_win_rate_pct"])
    return {
        "pick_targets": picks[:limit],
        "ban_targets": bans[:limit],
        "first_pick": picks[0] if picks else None,
        "first_ban": bans[0] if bans else None,
    }

def generate_team_insights(summary: dict, agents_agg: list, role_counts: Dict[str, int], players: list) -> list:
    insights = []

    def add(level, tag, text):
        insights.append({"level": level, "tag": tag, "text": text})

    total_games = int(summary.get("total_games", 0))
    if total_games < 20:
        add("pill-bad", "Sample", f"Low sample size ({total_games} games). Treat map win% as volatile.")
    elif total_games < 50:
        add("pill-mid", "Sample", f"Moderate sample size ({total_games} games). Trends can still swing.")
    else:
        add("pill-good", "Sample", f"Solid sample size ({total_games} games). Trends are more trustworthy.")

    stable_maps = int(summary.get("stable_map_count", 0))
    if stable_maps <= 3:
        add("pill-bad", "Map Pool", f"Shallow map pool ({stable_maps} maps with >=5 games). Expect bans to target comfort.")
    elif stable_maps <= 5:
        add("pill-mid", "Map Pool", f"Moderate map depth ({stable_maps} maps with >=5 games).")
    else:
        add("pill-good", "Map Pool", f"Wide map pool ({stable_maps} maps with >=5 games).")

    win_vol = float(summary.get("win_volatility", 0.0))
    if win_vol >= 12:
        add("pill-bad", "Consistency", f"High map volatility (win% stdev {win_vol:.1f}). Expect swingy results.")
    elif win_vol >= 7:
        add("pill-mid", "Consistency", f"Moderate map volatility (win% stdev {win_vol:.1f}).")
    else:
        add("pill-good", "Consistency", f"Stable across maps (win% stdev {win_vol:.1f}).")

    best_map = summary.get("best_map") or {}
    worst_map = summary.get("worst_map") or {}
    if best_map and worst_map and best_map.get("map") and worst_map.get("map"):
        best_games = int(best_map.get("games", 0))
        worst_games = int(worst_map.get("games", 0))
        if best_games >= 5 and worst_games >= 5 and best_map.get("map") != worst_map.get("map"):
            best_win = float(best_map.get("win_rate_pct", 0.0))
            worst_win = float(worst_map.get("win_rate_pct", 0.0))
            spread = best_win - worst_win
            if spread >= 20:
                add(
                    "pill-bad",
                    "Map Gap",
                    f"Large win-rate gap: {best_map['map']} {best_win:.1f}% vs {worst_map['map']} {worst_win:.1f}%. Expect bans into {worst_map['map']}.",
                )
            elif spread >= 12:
                add(
                    "pill-mid",
                    "Map Gap",
                    f"Noticeable map gap: {best_map['map']} {best_win:.1f}% vs {worst_map['map']} {worst_win:.1f}%.",
                )
            else:
                add(
                    "pill-good",
                    "Map Gap",
                    f"Map win rates are fairly tight (best {best_map['map']} {best_win:.1f}%, worst {worst_map['map']} {worst_win:.1f}%).",
                )

    avg_kd = float(summary.get("avg_kd", 0.0))
    avg_win = float(summary.get("avg_win_rate_pct", 0.0))
    if avg_kd >= 1.05 and avg_win < 50:
        add("pill-mid", "Conversion", "Fragging looks strong but win rate lags. Focus on mid-round trade and conversion.")
    elif avg_kd < 0.98 and avg_win >= 52:
        add("pill-good", "Teamplay", "Winning despite lower K/D. Utility, trades, and post-plants likely carry.")

    total_agent_matches = sum(int(a.get("matches", 0)) for a in agents_agg) or 0
    if total_agent_matches > 0:
        top2 = sum(int(a.get("matches", 0)) for a in agents_agg[:2])
        share = (top2 / total_agent_matches) * 100.0
        if share >= 60:
            add("pill-bad", "Agent Pool", f"Narrow agent pool (top 2 agents = {share:.0f}% of matches). Prep anti-strats.")
        elif share >= 45:
            add("pill-mid", "Agent Pool", f"Moderate agent reliance (top 2 = {share:.0f}% of matches).")
        else:
            add("pill-good", "Agent Pool", f"Healthy agent diversity (top 2 = {share:.0f}% of matches).")

        top_agent = agents_agg[0] if agents_agg else None
        if top_agent:
            top_share = (int(top_agent.get("matches", 0)) / total_agent_matches) * 100.0
            top_win = float(top_agent.get("win_rate", 0.0))
            diff = top_win - avg_win
            if top_share >= 30 and diff <= -4:
                add(
                    "pill-bad",
                    "Signature Agent",
                    f"Heavy reliance on {top_agent['agent']} ({top_share:.0f}% of matches) but win rate trails team average by {abs(diff):.1f} pts.",
                )
            elif top_share >= 30 and diff >= 4:
                add(
                    "pill-good",
                    "Signature Agent",
                    f"{top_agent['agent']} is a signature pick ({top_share:.0f}% of matches) and wins {diff:.1f} pts above team average.",
                )

        pocket = None
        for a in agents_agg:
            matches = int(a.get("matches", 0))
            if matches < 5:
                continue
            share = (matches / total_agent_matches) * 100.0
            if share > 12:
                continue
            win = float(a.get("win_rate", 0.0))
            diff = win - avg_win
            if diff >= 8:
                if not pocket or diff > pocket["diff"]:
                    pocket = {"agent": a.get("agent"), "share": share, "win": win, "diff": diff}
        if pocket:
            add(
                "pill-mid",
                "Pocket Pick",
                f"{pocket['agent']} wins {pocket['diff']:.1f} pts above team average on limited use ({pocket['share']:.0f}% of matches). Consider expanding if comp fits.",
            )

    if role_counts:
        total_role = sum(role_counts.values()) or 1
        top_role = max(role_counts.items(), key=lambda x: x[1])
        top_share = (top_role[1] / total_role) * 100.0
        if top_share >= 50:
            add("pill-mid", "Role Bias", f"{top_role[0]} heavy ({top_share:.0f}% of agent picks). Expect role-centric game plan.")

    # Impact concentration (top-2 players share)
    if players:
        sorted_players = sorted(players, key=lambda p: p.get("impact_index", 0.0), reverse=True)
        total_impact = sum(float(p.get("impact_index", 0.0)) for p in sorted_players) or 1.0
        top2 = sum(float(p.get("impact_index", 0.0)) for p in sorted_players[:2])
        share = (top2 / total_impact) * 100.0
        if share >= 55:
            add("pill-mid", "Reliance", f"Top-heavy impact (top 2 = {share:.0f}% of team impact). Target carries.")

    return insights

def summarize_team_snapshot(snap: dict, season_id: str) -> dict:
    """
    Returns a rich summary suitable for scouting + comparisons.
    """
    team_key = snap.get("team_key")
    team_name = snap.get("team_name") or team_key
    roster = snap.get("players") or []
    season_block = (snap.get("data") or {}).get(season_id, {}) or {}

    # --- Aggregate maps across roster ---
    players_maps = {}
    for rid in roster:
        entry = season_block.get(rid) or {}
        players_maps[rid] = entry.get("maps") or []

    team_agg = aggregate_team_maps(players_maps)  # your existing function
    map_rows = team_agg.get("by_map") or []

    agent_pools, role_counts_from_agents = build_agent_pools(season_block, roster)
    map_games_by_player = build_map_games_by_player(season_block, roster)

    total_games = sum(int(r.get("games", 0)) for r in map_rows)
    avg_win = _weighted_avg(map_rows, "win_rate_pct", "games")
    avg_kd = _weighted_avg(map_rows, "kd", "games")
    avg_dmg = _weighted_avg(map_rows, "dmg_round", "games")
    avg_score = _weighted_avg(map_rows, "score_round", "games")

    # Pool breadth: how many maps above N games
    map_min_games = 5
    stable_maps = [r for r in map_rows if int(r.get("games", 0)) >= map_min_games]
    stable_map_count = len(stable_maps)

    # Volatility: stdev of map win% weighted-ish by sample size filter
    win_rates = [float(r.get("win_rate_pct", 0.0)) for r in stable_maps]
    win_volatility = _stdev(win_rates)

    best_map = max(map_rows, key=lambda r: (r.get("win_rate_pct", 0.0), r.get("games", 0)), default=None)
    worst_map = min(map_rows, key=lambda r: (r.get("win_rate_pct", 999.0), -r.get("games", 0)), default=None)

    # --- Player-level rollups (from maps) ---
    players = []
    for rid in roster:
        entry = season_block.get(rid) or {}
        maps = entry.get("maps") or []
        wins = sum(int(m.get("wins", 0)) for m in maps)
        losses = sum(int(m.get("losses", 0)) for m in maps)
        games = wins + losses
        win = _safe_div(wins, games) * 100.0

        # weighted perf from maps
        w_kd = w_kr = w_dmg = w_score = 0.0
        gsum = 0
        for m in maps:
            g = int(m.get("wins", 0)) + int(m.get("losses", 0))
            if g <= 0:
                continue
            gsum += g
            w_kd += float(m.get("kd", 0.0)) * g
            w_kr += float(m.get("kill_round", 0.0)) * g
            w_dmg += float(m.get("dmg_round", 0.0)) * g
            w_score += float(m.get("score_round", 0.0)) * g

        players.append({
            "riot_id": rid,
            "games": games,
            "win_rate_pct": win,
            "kd": _safe_div(w_kd, gsum),
            "kill_round": _safe_div(w_kr, gsum),
            "dmg_round": _safe_div(w_dmg, gsum),
            "score_round": _safe_div(w_score, gsum),
        })

    players.sort(key=lambda p: (p["games"], p["score_round"]), reverse=True)

    # Carry index (simple scouting heuristic): score_round * kd * log(games+1)
    for p in players:
        p["impact_index"] = float(p["score_round"]) * max(0.5, float(p["kd"])) * math.log(p["games"] + 1.0)

    top_impact = sorted(players, key=lambda p: p["impact_index"], reverse=True)[:3]

    # --- Agents aggregation across roster ---
    agent_rows = []
    for rid, pool in agent_pools.items():
        for a in pool:
            agent_rows.append({
                "agent": a["agent"],
                "matches": a["matches"],
                "win_rate": a["win_rate"],
                "acs": a["acs"],
                "dmg_round": a["dmg_round"],
            })
    # aggregate by agent
    by_agent = {}
    for r in agent_rows:
        k = r["agent"]
        if k not in by_agent:
            by_agent[k] = {"agent": k, "matches": 0, "sum_win": 0.0, "sum_acs": 0.0, "sum_dmg": 0.0}
        by_agent[k]["matches"] += r["matches"]
        by_agent[k]["sum_win"] += r["win_rate"] * r["matches"]
        by_agent[k]["sum_acs"] += r["acs"] * r["matches"]
        by_agent[k]["sum_dmg"] += r["dmg_round"] * r["matches"]

    agents_agg = []
    for k, v in by_agent.items():
        m = v["matches"]
        agents_agg.append({
            "agent": k,
            "matches": m,
            "win_rate": _safe_div(v["sum_win"], m),
            "acs": _safe_div(v["sum_acs"], m),
            "dmg_round": _safe_div(v["sum_dmg"], m),
        })
    agents_agg.sort(key=lambda x: x["matches"], reverse=True)
    top_agents = agents_agg[:5]

    # Role balance from profile_stats roles_card if present (fallback)
    role_counts_profile = {}
    for rid in roster:
        entry = season_block.get(rid) or {}
        roles = (entry.get("profile_stats") or {}).get("roles_card") or []
        for rr in roles:
            role = rr.get("role")
            if not role:
                continue
            # role share might exist as "62%" in rr["rate"] or similar; if not, count appearances
    role_counts_profile[role] = role_counts_profile.get(role, 0) + 1

    role_counts = role_counts_from_agents or role_counts_profile

    team_summary = {
        "total_games": total_games,
        "avg_win_rate_pct": avg_win,
        "avg_kd": avg_kd,
        "avg_dmg_round": avg_dmg,
        "avg_score_round": avg_score,
        "stable_map_count": stable_map_count,
        "win_volatility": win_volatility,
        "best_map": best_map,
        "worst_map": worst_map,
    }
    top_agent = top_agents[0] if top_agents else None
    team_summary["top_agent"] = top_agent

    starting_roster = [p["riot_id"] for p in players[:5]]
    expected_comps = build_expected_comps(
        map_rows=map_rows,
        map_games_by_player=map_games_by_player,
        agent_pools=agent_pools,
        fallback_player_order=starting_roster,
    )

    map_plan = recommend_map_plan(map_rows)
    team_insights = generate_team_insights(team_summary, agents_agg, role_counts, players)

    return {
        "team_key": team_key,
        "team_name": team_name,
        "season_id": season_id,
        "roster": roster,
        "map_rows": map_rows,
        "players": players,
        "agents": agents_agg,
        "summary": team_summary,
        "top_impact_players": top_impact,
        "top_agents": top_agents,
        "role_balance": role_counts,
        "map_plan": map_plan,
        "expected_comps": expected_comps,
        "team_insights": team_insights,
    }

def generate_comparison_insights(A: dict, B: dict) -> list:
    """
    Returns list[ {level, tag, text} ] for templates.
    level: pill-good | pill-mid | pill-bad (reuse your CSS)
    """
    insights = []

    aS = A["summary"]
    bS = B["summary"]

    def add(level, tag, text):
        insights.append({"level": level, "tag": tag, "text": text})

    # Sample size / confidence
    min_games = min(aS["total_games"], bS["total_games"])
    if min_games < 20:
        add("pill-bad", "Confidence",
            f"Low sample size: {A['team_name']} has {aS['total_games']} games and {B['team_name']} has {bS['total_games']}. Treat win% as volatile.")
    elif min_games < 50:
        add("pill-mid", "Confidence",
            f"Moderate sample size: {A['team_name']} {aS['total_games']} games vs {B['team_name']} {bS['total_games']}. Win% still swings on map pool.")
    else:
        add("pill-good", "Confidence",
            f"Solid sample size: {A['team_name']} {aS['total_games']} games vs {B['team_name']} {bS['total_games']}. Trends are more trustworthy.")

    # Map pool breadth
    if aS["stable_map_count"] > bS["stable_map_count"] + 1:
        add("pill-good", "Map Pool", f"{A['team_name']} looks more battle-tested: {aS['stable_map_count']} maps with >=5 games vs {bS['stable_map_count']}.")
    elif bS["stable_map_count"] > aS["stable_map_count"] + 1:
        add("pill-good", "Map Pool", f"{B['team_name']} looks more battle-tested: {bS['stable_map_count']} maps with >=5 games vs {aS['stable_map_count']}.")
    else:
        add("pill-mid", "Map Pool", f"Similar map pool breadth (>=5 games): {A['team_name']} {aS['stable_map_count']} vs {B['team_name']} {bS['stable_map_count']}.")

    # Consistency (win volatility)
    if aS["win_volatility"] + 4 < bS["win_volatility"]:
        add("pill-good", "Consistency",
            f"{A['team_name']} is more consistent across maps (win% volatility {aS['win_volatility']:.1f} vs {bS['win_volatility']:.1f}).")
    elif bS["win_volatility"] + 4 < aS["win_volatility"]:
        add("pill-good", "Consistency",
            f"{B['team_name']} is more consistent across maps (win% volatility {bS['win_volatility']:.1f} vs {aS['win_volatility']:.1f}).")
    else:
        add("pill-mid", "Consistency",
            f"Map-to-map consistency is similar (volatility: {A['team_name']} {aS['win_volatility']:.1f} vs {B['team_name']} {bS['win_volatility']:.1f}).")

    # Core performance (weighted)
    def perf_delta(metric, pretty, good_if_positive=True):
        da = float(aS.get(metric, 0.0))
        db = float(bS.get(metric, 0.0))
        diff = da - db
        if abs(diff) < 1e-6:
            add("pill-mid", pretty, f"Even on {pretty.lower()} ({da:.2f} vs {db:.2f}).")
            return
        lead = A["team_name"] if diff > 0 else B["team_name"]
        valA = f"{da:.2f}" if "pct" not in metric else f"{da:.1f}%"
        valB = f"{db:.2f}" if "pct" not in metric else f"{db:.1f}%"
        mag = abs(diff)
        level = "pill-mid"
        if mag >= (5.0 if "pct" in metric else 0.10):
            level = "pill-good"
        add(level, pretty, f"{lead} leads on {pretty.lower()} ({valA} vs {valB}).")

    perf_delta("avg_win_rate_pct", "Win Rate")
    perf_delta("avg_kd", "K/D")
    perf_delta("avg_dmg_round", "DMG/R")
    perf_delta("avg_score_round", "Score/R")

    # Map-specific head-to-head: biggest win% gaps on shared maps
    a_by_map = {r.get("map"): r for r in A["map_rows"]}
    b_by_map = {r.get("map"): r for r in B["map_rows"]}
    shared = [m for m in a_by_map.keys() if m in b_by_map and m]
    deltas = []
    for m in shared:
        ra, rb = a_by_map[m], b_by_map[m]
        ga, gb = int(ra.get("games", 0)), int(rb.get("games", 0))
        if ga < 3 or gb < 3:
            continue
        deltas.append((m, float(ra.get("win_rate_pct", 0.0)) - float(rb.get("win_rate_pct", 0.0)), ga, gb))
    deltas.sort(key=lambda x: abs(x[1]), reverse=True)

    if deltas:
        m, d, ga, gb = deltas[0]
        lead = A["team_name"] if d > 0 else B["team_name"]
        add("pill-good", "Map Edge",
            f"Biggest shared-map edge: {lead} on {m} (win% gap {abs(d):.1f} pts; samples {ga}g vs {gb}g).")
    else:
        add("pill-mid", "Map Edge",
            "No strong shared-map advantage found (or too few games on shared maps).")

    # Pick/Ban suggestions using adjusted win rates on shared maps
    def map_advantage_recos(team_a: dict, team_b: dict):
        a_strength = {r["map"]: r for r in compute_map_strengths(team_a["map_rows"])}
        b_strength = {r["map"]: r for r in compute_map_strengths(team_b["map_rows"])}
        shared_maps = [m for m in a_strength.keys() if m in b_strength and m]
        rows = []
        for m in shared_maps:
            ra, rb = a_strength[m], b_strength[m]
            ga, gb = int(ra.get("games", 0)), int(rb.get("games", 0))
            if min(ga, gb) < 3:
                continue
            adv = float(ra.get("adj_win_rate_pct", 0.0)) - float(rb.get("adj_win_rate_pct", 0.0))
            rows.append((m, adv, ra, rb))
        if not rows:
            return None
        rows.sort(key=lambda x: x[1], reverse=True)
        pick = rows[0]
        ban = rows[-1]
        return {"pick": pick, "ban": ban}

    recos = map_advantage_recos(A, B)
    if recos:
        m, adv, ra, rb = recos["pick"]
        add("pill-good", f"{A['team_name']} Pick",
            f"Suggested pick: {m} (adj {ra['adj_win_rate_pct']:.1f}% vs {rb['adj_win_rate_pct']:.1f}%).")
        m, adv, ra, rb = recos["ban"]
        add("pill-bad", f"{A['team_name']} Ban",
            f"Suggested ban: {m} (adj {ra['adj_win_rate_pct']:.1f}% vs {rb['adj_win_rate_pct']:.1f}%).")

    recos_b = map_advantage_recos(B, A)
    if recos_b:
        m, adv, rb, ra = recos_b["pick"]
        add("pill-good", f"{B['team_name']} Pick",
            f"Suggested pick: {m} (adj {rb['adj_win_rate_pct']:.1f}% vs {ra['adj_win_rate_pct']:.1f}%).")
        m, adv, rb, ra = recos_b["ban"]
        add("pill-bad", f"{B['team_name']} Ban",
            f"Suggested ban: {m} (adj {rb['adj_win_rate_pct']:.1f}% vs {ra['adj_win_rate_pct']:.1f}%).")

    # Player impact concentration: is one team carried by 1-2 players?
    def impact_concentration(team):
        ps = team["players"]
        top = sorted(ps, key=lambda p: p.get("impact_index", 0.0), reverse=True)
        if not top:
            return 0.0
        total = sum(p.get("impact_index", 0.0) for p in top) or 1.0
        top2 = sum(p.get("impact_index", 0.0) for p in top[:2])
        return (top2 / total) * 100.0

    a_conc = impact_concentration(A)
    b_conc = impact_concentration(B)
    if a_conc > b_conc + 10:
        add("pill-mid", "Reliance",
            f"{A['team_name']} is more top-heavy (top-2 impact share ~{a_conc:.0f}% vs {b_conc:.0f}%). Target their stars to swing rounds.")
    elif b_conc > a_conc + 10:
        add("pill-mid", "Reliance",
            f"{B['team_name']} is more top-heavy (top-2 impact share ~{b_conc:.0f}% vs {a_conc:.0f}%). Target their stars to swing rounds.")
    else:
        add("pill-good", "Reliance",
            f"Similar impact distribution (top-2 share: {A['team_name']} ~{a_conc:.0f}%, {B['team_name']} ~{b_conc:.0f}%).")

    # Agent identity: top agents overlap?
    a_agents = [x["agent"] for x in A.get("top_agents", [])]
    b_agents = [x["agent"] for x in B.get("top_agents", [])]
    overlap = [x for x in a_agents if x in b_agents]
    if overlap:
        add("pill-mid", "Agent Meta",
            f"Both teams share key agent comfort picks: {', '.join(overlap[:3])}{'...' if len(overlap)>3 else ''}.")
    else:
        add("pill-good", "Agent Meta",
            "Top agent pools differ - higher chance of comp mismatch and exploitable utility gaps.")

    return insights

def save_comparison_dashboard(A_snap: dict, B_snap: dict, season_id: str, A_summary: dict, B_summary: dict, insights: list) -> dict:
    """
    Persist a dashboard JSON on disk. Returns metadata.
    """
    base = f"{A_summary['team_key']}|{B_summary['team_key']}|{season_id}|{_now_iso()}"
    dash_id = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    payload = {
        "dash_id": dash_id,
        "created_ts": _now_iso(),
        "season_id": season_id,
        "team_a_key": A_summary["team_key"],
        "team_b_key": B_summary["team_key"],
        "team_a_name": A_summary["team_name"],
        "team_b_name": B_summary["team_name"],
        "A": A_summary,
        "B": B_summary,
        "insights": insights,
    }
    (DASHBOARD_DIR / f"{dash_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
# ============================================================
# Team file parsing (first line = team name)
# ============================================================
def parse_team_upload(text: str) -> Tuple[str, List[str]]:
    """
    First non-empty line = team name
    Remaining lines = riot IDs in format name#tag
    """
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("//") and not ln.startswith("# ")]
    if not lines:
        return ("Unnamed Team", [])

    team_name = lines[0].strip()
    ids: List[str] = []
    for raw in lines[1:]:
        line = raw.strip()
        if line.count("#") != 1:
            continue
        name, tag = line.split("#", 1)
        name, tag = name.strip(), tag.strip()
        if not name or not tag:
            continue
        ids.append(f"{name}#{tag}")

    # unique preserve order
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return (team_name, out)


def team_key_from_ids(team_name: str, team_ids: List[str]) -> str:
    base = team_name.strip() + "-" + "-".join(team_ids)
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", base)[:120] or "team"


# ============================================================
# URL builders
# ============================================================
def riot_to_opgg_slug(riot_id: str) -> str:
    name, tag = riot_id.split("#", 1)
    return f"{name}-{tag}"


def opgg_profile_url_from_slug(slug: str, season_id: str) -> str:
    return f"https://op.gg/valorant/profile/{slug}?seasonId={season_id}"


def opgg_maps_url_from_slug(slug: str, season_id: str) -> str:
    return f"https://op.gg/valorant/profile/{slug}/maps?seasonId={season_id}"


def opgg_agents_url_from_slug(slug: str, season_id: str) -> str:
    return f"https://op.gg/valorant/profile/{slug}/agents?seasonId={season_id}"


# ============================================================
# Parsing helpers
# ============================================================
def text_clean(el) -> str:
    if not el:
        return ""
    return el.get_text(" ", strip=True)


def to_float(s: str) -> float:
    if s is None:
        return 0.0
    s = str(s).strip().replace("%", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0


def to_int(s: str) -> int:
    if s is None:
        return 0
    s = str(s).strip().replace(",", "")
    try:
        return int(float(s))
    except Exception:
        return 0


# ============================================================
# Scrapers (Playwright)
# ============================================================
def _goto_with_delay(page, url: str, extra_wait_ms: int = 5000) -> None:
    page.goto(url, wait_until="domcontentloaded")
    # your request: always give it time to hydrate
    page.wait_for_timeout(extra_wait_ms)


def scrape_maps_page(context, url: str, debug_prefix: str) -> List[Dict[str, Any]]:
    page = context.new_page()
    try:
        _goto_with_delay(page, url, extra_wait_ms=5000)
        # lazy render assist
        page.mouse.wheel(0, 1200)
        page.wait_for_timeout(1200)

        # wait for any table
        selectors = [
            "table tbody tr",
            "table",
            "th:has-text('Maps')",
            "button:has-text('Win rate')",
        ]
        last_err = None
        for sel in selectors:
            try:
                page.wait_for_selector(sel, timeout=25_000)
                break
            except PlaywrightTimeoutError as e:
                last_err = e
        else:
            Path(f"{debug_prefix}.html").write_text(page.content(), encoding="utf-8")
            page.screenshot(path=f"{debug_prefix}.png", full_page=True)
            raise RuntimeError(
                f"Timed out waiting for OP.GG maps table. Saved {debug_prefix}.png/.html"
            ) from last_err

        html = page.content()
    finally:
        try:
            page.close()
        except Exception:
            pass

    soup = BeautifulSoup(html, "html.parser")

    # find maps table robustly by headers text
    table = None
    for t in soup.find_all("table"):
        hdr = t.get_text(" ", strip=True)
        if "Maps" in hdr and ("Win rate" in hdr or "Win" in hdr) and ("Dmg/Round" in hdr or "Dmg/R" in hdr):
            table = t
            break
    if table is None:
        Path(f"{debug_prefix}_parsed.html").write_text(html, encoding="utf-8")
        raise RuntimeError(f"Maps table not found. Saved {debug_prefix}_parsed.html")

    rows: List[Dict[str, Any]] = []
    for tr in table.select("tbody tr"):
        if "ad" in (tr.get("class") or []):
            continue
        tds = tr.find_all("td")
        if len(tds) < 8:
            continue

        map_span = tds[0].select_one("span.truncate")
        map_name = map_span.get_text(strip=True) if map_span else text_clean(tds[0])

        def clean(i: int) -> str:
            return text_clean(tds[i])

        rows.append(
            {
                "map": map_name,
                "win_rate": clean(1),
                "wins": clean(2),
                "losses": clean(3),
                "kd": clean(4),
                "kill_round": clean(5),
                "dmg_round": clean(6),
                "score_round": clean(7),
            }
        )

    return rows


def scrape_agents_page(context, url: str, debug_prefix: str) -> List[Dict[str, Any]]:
    page = context.new_page()
    try:
        _goto_with_delay(page, url, extra_wait_ms=5000)
        page.mouse.wheel(0, 1200)
        page.wait_for_timeout(1000)

        # basic wait: any table or list rows
        selectors = [
            "table tbody tr",
            "tbody tr",
            "main",
        ]
        for sel in selectors:
            try:
                page.wait_for_selector(sel, timeout=25_000)
                break
            except PlaywrightTimeoutError:
                continue

        html = page.content()
    finally:
        try:
            page.close()
        except Exception:
            pass

    soup = BeautifulSoup(html, "html.parser")

    # Heuristic: agent rows usually have stat columns like
    # Agent | Matches | Win% | KDA | K/D | ACS | Dmg/R
    agents: List[Dict[str, Any]] = []

    for t in soup.find_all("table"):
        rows = t.select("tbody tr")
        if not rows:
            continue

        for tr in rows:
            tds = tr.find_all(["td", "th"])
            if len(tds) < 3:
                continue

            cells = [text_clean(td) for td in tds]
            if len(cells) < 3:
                continue

            # Expect matches in col 2 and a % in col 3.
            if "%" not in cells[2] or not re.search(r"\d", cells[1]):
                continue

            agent_name = cells[0]
            if not agent_name or len(agent_name) > 60:
                continue

            win_rate_pct = to_float(cells[2])
            kda = None
            if len(cells) >= 4:
                m = re.search(r"(\d+(?:\.\d+)?)\s*:\s*1", cells[3])
                if m:
                    kda = f"{m.group(1)}:1"
                elif cells[3]:
                    kda = cells[3]

            agents.append(
                {
                    "agent": agent_name,
                    "matches": to_int(cells[1]),
                    "win_rate": win_rate_pct,
                    "win_rate_pct": win_rate_pct,
                    "kda": kda,
                    "kd": to_float(cells[4]) if len(cells) > 4 else None,
                    "acs": to_float(cells[5]) if len(cells) > 5 else None,
                    "dmg_round": to_float(cells[6]) if len(cells) > 6 else None,
                    "raw": " ".join([c for c in cells if c]),
                }
            )

        if agents:
            break

    if agents:
        return agents

    # Fallback: best-effort parse from row text if the table layout changed.
    for t in soup.find_all("table"):
        for tr in t.select("tbody tr"):
            row_text = text_clean(tr)
            if not row_text:
                continue
            # crude filters to avoid unrelated tables
            if "map" in row_text.lower():
                continue

            m = re.search(r"(\d+)\s+(\d+(?:\.\d+)?)%", row_text)
            if not m:
                continue

            agent_name = row_text.split(m.group(0))[0].strip()
            if not agent_name or len(agent_name) > 60:
                continue

            kda = None
            m_kda = re.search(r"(\d+(?:\.\d+)?)\s*:\s*1", row_text)
            if m_kda:
                kda = f"{m_kda.group(1)}:1"

            nums = [to_float(n) for n in re.findall(r"\d+(?:\.\d+)?", row_text.replace(",", ""))]
            kd = nums[-3] if len(nums) >= 3 else None
            acs = nums[-2] if len(nums) >= 2 else None
            dmg_round = nums[-1] if len(nums) >= 1 else None

            agents.append(
                {
                    "agent": agent_name,
                    "matches": to_int(m.group(1)),
                    "win_rate": to_float(m.group(2)),
                    "win_rate_pct": to_float(m.group(2)),
                    "kda": kda,
                    "kd": kd,
                    "acs": acs,
                    "dmg_round": dmg_round,
                    "raw": row_text,
                }
            )

        if agents:
            break

    return agents


def parse_profile_stats_and_roles(html: str) -> Dict[str, Any]:
    """
    Parses the "Stats" card (top metrics + breakdown) and "Roles" card.
    Avoids Tailwind bracket class selectors that break soupsieve.
    """
    soup = BeautifulSoup(html, "html.parser")
    out: Dict[str, Any] = {"stats_card": {"top": [], "breakdown": []}, "roles_card": []}

    # Find "Stats" card by h3 text
    stats_card = None
    for h3 in soup.find_all(["h3", "h2"]):
        if h3.get_text(strip=True).lower() == "stats":
            stats_card = h3.find_parent("div")
            break
    if stats_card is None:
        # Fallback for newer OP.GG layout: card with Dmg/round + KDA rows
        best = None
        best_len = 10**9
        for div in soup.find_all("div"):
            txt = div.get_text(" ", strip=True)
            if "Stats" in txt and "Dmg/round" in txt and "KDA" in txt:
                l = len(txt)
                if l < best_len:
                    best_len = l
                    best = div
        stats_card = best

    if stats_card:
        # In older layout: first UL has 6 items; second UL is breakdown.
        uls = stats_card.find_all("ul")
        if len(uls) >= 1:
            items = []
            for li in uls[0].find_all("li"):
                spans = li.find_all("span")
                if len(spans) >= 2:
                    label = spans[0].get_text(" ", strip=True)
                    value = spans[1].get_text(" ", strip=True)
                    items.append({"label": label, "value": value})
            out["stats_card"]["top"] = items

        if len(uls) >= 2:
            items2 = []
            for li in uls[1].find_all("li"):
                spans = li.find_all("span")
                if len(spans) >= 2:
                    label = spans[0].get_text(" ", strip=True)
                    value = spans[1].get_text(" ", strip=True)
                    items2.append({"label": label, "value": value})
            out["stats_card"]["breakdown"] = items2

        if not out["stats_card"]["top"] and not out["stats_card"]["breakdown"]:
            # Fallback: gather all li rows in the card.
            items = []
            for li in stats_card.find_all("li"):
                spans = li.find_all("span")
                if len(spans) >= 2:
                    label = spans[0].get_text(" ", strip=True)
                    value = spans[1].get_text(" ", strip=True)
                    if label and value:
                        items.append({"label": label, "value": value})
            if items:
                if len(items) > 6:
                    out["stats_card"]["top"] = items[:6]
                    out["stats_card"]["breakdown"] = items[6:]
                else:
                    out["stats_card"]["top"] = items

    if not out["stats_card"]["top"] and not out["stats_card"]["breakdown"]:
        # If we latched onto a menu header, retry with content-based search.
        best = None
        best_len = 10**9
        for div in soup.find_all("div"):
            txt = div.get_text(" ", strip=True)
            if "Stats" in txt and "Dmg/round" in txt and "KDA" in txt:
                l = len(txt)
                if l < best_len:
                    best_len = l
                    best = div
        if best:
            items = []
            for li in best.find_all("li"):
                spans = li.find_all("span")
                if len(spans) >= 2:
                    label = spans[0].get_text(" ", strip=True)
                    value = spans[1].get_text(" ", strip=True)
                    if label and value:
                        items.append({"label": label, "value": value})
            if items:
                if len(items) > 6:
                    out["stats_card"]["top"] = items[:6]
                    out["stats_card"]["breakdown"] = items[6:]
                else:
                    out["stats_card"]["top"] = items

    # Find "Roles" card by h3 text
    roles_card = None
    for h3 in soup.find_all(["h3", "h2"]):
        if h3.get_text(strip=True).lower() == "roles":
            roles_card = h3.find_parent("div")
            break
    if roles_card is None:
        # Fallback for newer OP.GG layout
        best = None
        best_len = 10**9
        for div in soup.find_all("div"):
            txt = div.get_text(" ", strip=True)
            if txt.startswith("Roles") and "KDA" in txt and "W / L" in txt:
                l = len(txt)
                if l < best_len:
                    best_len = l
                    best = div
        if best is None:
            for div in soup.find_all("div"):
                txt = div.get_text(" ", strip=True)
                if txt.startswith("Roles") and "KDA" in txt:
                    l = len(txt)
                    if l < best_len:
                        best_len = l
                        best = div
        roles_card = best

    roles: List[Dict[str, Any]] = []
    def _parse_roles(card):
        parsed: List[Dict[str, Any]] = []
        if not card:
            return parsed
        for li in card.find_all("li"):
            row_text = li.get_text(" ", strip=True)
            if not row_text:
                continue

            # Role name: usually the first bold-ish label
            role_name = ""
            for cand in ["Sentinel", "Duelist", "Initiator", "Controller"]:
                if cand.lower() in row_text.lower():
                    role_name = cand
                    break
            if not role_name:
                continue

            # KDA appears like "1.38:1 KDA"
            m_kda = re.search(r"(\d+(?:\.\d+)?\s*:\s*\d+)\s*KDA", row_text, re.IGNORECASE)
            kda = m_kda.group(1).replace(" ", "") if m_kda else ""

            # K/D/A counts: "185 / 171 / 51"
            m_kda_counts = re.search(r"(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", row_text)
            kills = int(m_kda_counts.group(1)) if m_kda_counts else None
            deaths = int(m_kda_counts.group(2)) if m_kda_counts else None
            assists = int(m_kda_counts.group(3)) if m_kda_counts else None

            # W/L: "10W / 3L"
            m_wl = re.search(r"(\d+)\s*W\s*/\s*(\d+)\s*L", row_text, re.IGNORECASE)
            wins = int(m_wl.group(1)) if m_wl else None
            losses = int(m_wl.group(2)) if m_wl else None
            win_rate = ""
            m_wr = re.search(r"(\d+)%\s*\d+\s*W\s*/\s*\d+\s*L", row_text, re.IGNORECASE)
            if m_wr:
                win_rate = m_wr.group(1) + "%"

            # Role %: "62%" near role label line
            role_pct = ""
            try:
                idx = row_text.lower().index(role_name.lower())
                tail = row_text[idx:]
                m_pct = re.search(r"(\d+)%", tail)
                role_pct = (m_pct.group(1) + "%") if m_pct else ""
            except Exception:
                role_pct = ""

            parsed.append(
                {
                    "role": role_name,
                    "role_pct": role_pct,
                    "kda": kda,
                    "kills": kills,
                    "deaths": deaths,
                    "assists": assists,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": win_rate,
                }
            )
        return parsed

    roles = _parse_roles(roles_card)
    if not roles:
        # Retry with content-based search if header-based selection failed.
        best = None
        best_len = 10**9
        for div in soup.find_all("div"):
            txt = div.get_text(" ", strip=True)
            if txt.startswith("Roles") and "KDA" in txt:
                l = len(txt)
                if l < best_len:
                    best_len = l
                    best = div
        roles = _parse_roles(best)

    out["roles_card"] = roles
    return out


def scrape_profile_page(
    context,
    url: str,
    debug_prefix: str,
) -> Dict[str, Any]:
    page = context.new_page()
    try:
        _goto_with_delay(page, url, extra_wait_ms=5000)
        page.mouse.wheel(0, 1200)
        page.wait_for_timeout(1200)

        # Don’t hard-fail on exact selector; just try to ensure content exists
        try:
            page.wait_for_selector("main", timeout=20_000)
        except PlaywrightTimeoutError:
            pass

        html = page.content()
    finally:
        try:
            page.close()
        except Exception:
            pass

    try:
        return parse_profile_stats_and_roles(html)
    except Exception as e:
        Path(f"{debug_prefix}.html").write_text(html, encoding="utf-8")
        raise RuntimeError(f"Failed parsing profile page. Saved {debug_prefix}.html") from e


def normalize_maps(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        out.append(
            {
                "map": r.get("map", ""),
                "win_rate_pct": to_float(r.get("win_rate", "")),
                "wins": to_int(r.get("wins", "")),
                "losses": to_int(r.get("losses", "")),
                "kd": to_float(r.get("kd", "")),
                "kill_round": to_float(r.get("kill_round", "")),
                "dmg_round": to_float(r.get("dmg_round", "")),
                "score_round": to_float(r.get("score_round", "")),
                "raw": r,
            }
        )
    return out


def normalize_agents(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        agent = r.get("agent") or r.get("name") or ""
        raw = r.get("raw") or ""

        matches = r.get("matches")
        win_rate_pct = r.get("win_rate_pct")
        win_rate = r.get("win_rate")
        kda = r.get("kda")
        kd = r.get("kd")
        acs = r.get("acs")
        dmg_round = r.get("dmg_round")

        if (matches in (None, "", 0)) or win_rate_pct is None:
            if raw:
                m = re.search(r"(\d+)\s+(\d+(?:\.\d+)?)%", raw)
                if m:
                    if matches in (None, "", 0):
                        matches = to_int(m.group(1))
                    if win_rate_pct is None:
                        win_rate_pct = to_float(m.group(2))

        if win_rate_pct is None:
            win_rate_pct = to_float(win_rate)

        matches = to_int(matches)
        win_rate_pct = to_float(win_rate_pct)

        if not kda and raw:
            m = re.search(r"(\d+(?:\.\d+)?)\s*:\s*1", raw)
            if m:
                kda = f"{m.group(1)}:1"

        if raw and (kd in (None, "") or acs in (None, "") or dmg_round in (None, "")):
            nums = [to_float(n) for n in re.findall(r"\d+(?:\.\d+)?", raw.replace(",", ""))]
            if len(nums) >= 3:
                if kd in (None, ""):
                    kd = nums[-3]
                if acs in (None, ""):
                    acs = nums[-2]
                if dmg_round in (None, ""):
                    dmg_round = nums[-1]

        out.append(
            {
                "agent": agent,
                "matches": matches,
                "win_rate": win_rate_pct,
                "win_rate_pct": win_rate_pct,
                "kda": kda,
                "kd": kd,
                "acs": acs,
                "dmg_round": dmg_round,
                "raw": raw,
            }
        )

    return out


# ============================================================
# Aggregation / Insights
# ============================================================
def aggregate_team_maps(players_maps: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    by_map: Dict[str, Dict[str, Any]] = {}

    for riot_id, rows in players_maps.items():
        for r in rows:
            m = r.get("map") or ""
            if not m:
                continue
            games = int(r.get("wins", 0)) + int(r.get("losses", 0))
            if games <= 0:
                continue

            if m not in by_map:
                by_map[m] = {
                    "map": m,
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "sum_kd": 0.0,
                    "sum_kr": 0.0,
                    "sum_dmg": 0.0,
                    "sum_score": 0.0,
                }

            by_map[m]["games"] += games
            by_map[m]["wins"] += int(r.get("wins", 0))
            by_map[m]["losses"] += int(r.get("losses", 0))

            by_map[m]["sum_kd"] += float(r.get("kd", 0.0)) * games
            by_map[m]["sum_kr"] += float(r.get("kill_round", 0.0)) * games
            by_map[m]["sum_dmg"] += float(r.get("dmg_round", 0.0)) * games
            by_map[m]["sum_score"] += float(r.get("score_round", 0.0)) * games

    rows = []
    for m, v in by_map.items():
        g = v["games"]
        rows.append(
            {
                "map": m,
                "games": g,
                "wins": v["wins"],
                "losses": v["losses"],
                "win_rate_pct": (v["wins"] / g * 100.0) if g else 0.0,
                "kd": (v["sum_kd"] / g) if g else 0.0,
                "kill_round": (v["sum_kr"] / g) if g else 0.0,
                "dmg_round": (v["sum_dmg"] / g) if g else 0.0,
                "score_round": (v["sum_score"] / g) if g else 0.0,
            }
        )

    rows.sort(key=lambda x: x["games"], reverse=True)
    return {"by_map": rows}


# ============================================================
# Progress task system (SSE)
# ============================================================
@dataclass
class TaskState:
    pct: int = 0
    msg: str = "Queued..."
    done: bool = False
    error: Optional[str] = None
    redirect: Optional[str] = None


TASKS: Dict[str, TaskState] = {}


def _new_task_id() -> str:
    return f"t{int(time.time() * 1000)}_{os.getpid()}_{len(TASKS)+1}"


def set_task(task_id: str, pct: int, msg: str) -> None:
    st = TASKS.get(task_id)
    if not st:
        return
    st.pct = max(0, min(100, int(pct)))
    st.msg = msg


def finish_task(task_id: str, redirect_url: str) -> None:
    st = TASKS.get(task_id)
    if not st:
        return
    st.pct = 100
    st.msg = "Done."
    st.done = True
    st.redirect = redirect_url


def fail_task(task_id: str, err: str) -> None:
    st = TASKS.get(task_id)
    if not st:
        return
    st.error = err
    st.msg = err
    st.done = True
    st.pct = max(st.pct, 5)


# ============================================================
# Snapshot Builder (ALL players x ALL seasons)
# ============================================================
def build_team_snapshot(
    team_key: str,
    team_name: str,
    team_ids: List[str],
    seasons: Optional[Dict[str, str]] = None,
    polite_sleep: float = 0.8,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> dict:
    seasons = seasons or SEASONS
    total_steps = max(1, len(team_ids) * len(seasons) * 3)  # profile+maps+agents
    done_steps = 0

    def bump(msg: str):
        nonlocal done_steps
        done_steps += 1
        pct = int((done_steps / total_steps) * 90) + 5  # 5..95 during work
        if progress_cb:
            progress_cb(pct, msg)

    snap: Dict[str, Any] = {
        "team_key": team_key,
        "team_name": team_name,
        "updated_ts": int(time.time()),
        "players": team_ids,
        "seasons": seasons,
        "data": {},
    }

    with sync_playwright() as p:
        browser = None
        context = None
        try:
            browser = p.chromium.launch(
                headless=False,
                args=["--start-maximized", "--disable-blink-features=AutomationControlled"],
            )
            context = browser.new_context(viewport=None, locale="en-US")
            context.set_default_timeout(40_000)

            for season_label, season_id in seasons.items():
                snap["data"][season_id] = {}

                for riot_id in team_ids:
                    slug = riot_to_opgg_slug(riot_id)
                    profile_url = opgg_profile_url_from_slug(slug, season_id)
                    maps_url = opgg_maps_url_from_slug(slug, season_id)
                    agents_url = opgg_agents_url_from_slug(slug, season_id)

                    pref = f"{team_key}_{riot_id.replace('#','_')}_{season_id}"

                    bump(f"[{season_label}] {riot_id}: profile...")
                    profile_stats = scrape_profile_page(
                        context,
                        profile_url,
                        debug_prefix=f"profile_{pref}",
                    )

                    bump(f"[{season_label}] {riot_id}: maps...")
                    maps_raw = scrape_maps_page(
                        context,
                        maps_url,
                        debug_prefix=f"maps_{pref}",
                    )
                    maps = normalize_maps(maps_raw)

                    bump(f"[{season_label}] {riot_id}: agents...")
                    agents = scrape_agents_page(
                        context,
                        agents_url,
                        debug_prefix=f"agents_{pref}",
                    )
                    agents = normalize_agents(agents)

                    snap["data"][season_id][riot_id] = {
                        "season_label": season_label,
                        "season_id": season_id,
                        "riot_id": riot_id,
                        "slug": slug,
                        "profile_url": profile_url,
                        "maps_url": maps_url,
                        "agents_url": agents_url,
                        "profile_stats": profile_stats,
                        "maps": maps,
                        "agents": agents,
                    }

                    time.sleep(polite_sleep)

        finally:
            try:
                if context:
                    context.close()
            except Exception:
                pass
            try:
                if browser:
                    browser.close()
            except Exception:
                pass

    return snap


DASHBOARD_DIR = APP_DIR / "data" / "dashboards"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

def dashboard_path(dash_id: str) -> Path:
    return DASHBOARD_DIR / f"{dash_id}.json"

def load_dashboard(dash_id: str) -> dict:
    p = dashboard_path(dash_id)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_dashboard(dash_id: str, data: dict) -> None:
    dashboard_path(dash_id).write_text(json.dumps(data, indent=2), encoding="utf-8")

# ============================================================
# Routes
# ============================================================
@app.context_processor
def inject_globals():
    return {
        "seasons": SEASONS,
        "default_season": DEFAULT_SEASON_ID,
    }


@app.errorhandler(400)
def bad_request(error):
    message = getattr(error, "description", "Bad request.")
    return render_template(
        "error.html",
        code=400,
        title="Bad Request",
        message=message,
    ), 400


def _resolve_season_id(requested: str, snap: dict) -> str:
    data = snap.get("data") or {}
    if requested in data:
        return requested
    if data:
        return next(iter(data.keys()))
    return DEFAULT_SEASON_ID
@app.get("/")
def index():
    q = (request.args.get("q") or "").strip().lower()
    teams = list_snapshots()

    if q:
        teams = [
            t for t in teams
            if q in (t.get("team_name", "").lower()) or q in (t.get("team_key", "").lower())
        ]

    return render_template(
        "index.html",
        teams=teams,
        q=request.args.get("q") or "",
        seasons=SEASONS,
        default_season=DEFAULT_SEASON_ID,
    )


@app.post("/upload")
def upload():
    """
    Non-JS fallback: build synchronously (visible Chromium).
    Your index.html calls showProgress() on click; live updates require /upload_async + SSE.
    """
    if "file" not in request.files:
        return redirect(url_for("index"))

    f = request.files["file"]
    if not f or not f.filename:
        return redirect(url_for("index"))

    text = f.read().decode("utf-8", errors="replace")
    team_name, team_ids = parse_team_upload(text)
    if not team_ids:
        abort(400, "No valid riot IDs found. Use first line team name, then name#tag lines.")

    season_ids = request.form.getlist("season_id")
    season_ids = [sid for sid in season_ids if sid in SEASONS.values()]
    if not season_ids:
        abort(400, "Select at least one season to scrape.")
    seasons = {label: sid for label, sid in SEASONS.items() if sid in season_ids}
    default_season_id = next(iter(seasons.values()))

    team_key = team_key_from_ids(team_name, team_ids)
    (UPLOAD_DIR / f"{team_key}.txt").write_text("\n".join([team_name] + team_ids), encoding="utf-8")

    snap = build_team_snapshot(team_key, team_name, team_ids, seasons=seasons, polite_sleep=0.8)
    save_snapshot(team_key, snap)

    return redirect(url_for("team_view", team_key=team_key, seasonId=default_season_id))


@app.post("/upload_async")
def upload_async():
    """
    JS/SSE-friendly: start background scraper and return task_id + redirect.
    """
    if "file" not in request.files:
        abort(400, "Missing file")
    f = request.files["file"]
    if not f or not f.filename:
        abort(400, "Missing filename")

    text = f.read().decode("utf-8", errors="replace")
    team_name, team_ids = parse_team_upload(text)
    if not team_ids:
        abort(400, "No valid riot IDs found. Use one per line like name#tag (first line team name).")

    season_ids = request.form.getlist("season_id")
    season_ids = [sid for sid in season_ids if sid in SEASONS.values()]
    if not season_ids:
        abort(400, "Select at least one season to scrape.")
    seasons = {label: sid for label, sid in SEASONS.items() if sid in season_ids}
    default_season_id = next(iter(seasons.values()))

    team_key = team_key_from_ids(team_name, team_ids)
    (UPLOAD_DIR / f"{team_key}.txt").write_text("\n".join([team_name] + team_ids), encoding="utf-8")

    task_id = _new_task_id()
    TASKS[task_id] = TaskState(pct=2, msg="Starting...")

    def worker():
        try:
            set_task(task_id, 5, "Launching Chromium...")
            snap = build_team_snapshot(
                team_key,
                team_name,
                team_ids,
                seasons=seasons,
                polite_sleep=0.8,
                progress_cb=lambda pct, msg: set_task(task_id, pct, msg),
            )
            save_snapshot(team_key, snap)
            finish_task(task_id, url_for("team_view", team_key=team_key, seasonId=default_season_id))
        except Exception as e:
            fail_task(task_id, str(e))

    threading.Thread(target=worker, daemon=True).start()

    return jsonify(
        {
            "task_id": task_id,
            "redirect": url_for("team_view", team_key=team_key, seasonId=default_season_id),
        }
    )


@app.get("/progress/<task_id>")
def progress(task_id: str):
    """
    Server-Sent Events stream: emits {"pct":..,"msg":..,"done":..,"redirect":..}
    """
    if task_id not in TASKS:
        abort(404)

    def gen():
        last = None
        while True:
            st = TASKS.get(task_id)
            if not st:
                break

            payload = {
                "pct": st.pct,
                "msg": st.msg,
                "done": st.done,
                "redirect": st.redirect,
                "error": st.error,
            }
            txt = json.dumps(payload)
            if txt != last:
                yield f"data: {txt}\n\n"
                last = txt

            if st.done:
                break
            time.sleep(0.25)

    return app.response_class(gen(), mimetype="text/event-stream")


@app.get("/team/<team_key>")
def team_view(team_key: str):
    snap = load_snapshot(team_key)
    if not snap:
        abort(404)

    season_id = _resolve_season_id(request.args.get("seasonId") or DEFAULT_SEASON_ID, snap)

    team_name = snap.get("team_name") or team_key
    team_ids = snap.get("players") or []
    season_block = (snap.get("data") or {}).get(season_id, {})
    summary = summarize_team_snapshot(snap, season_id)
    map_rows = summary.get("map_rows") or []
    team_summary = summary.get("summary") or {}
    map_plan = summary.get("map_plan") or {}
    expected_comps = summary.get("expected_comps") or []
    team_insights = summary.get("team_insights") or []

    # Per-player summaries (roster table)
    summaries = []
    for rid in team_ids:
        entry = season_block.get(rid) or {}
        maps = entry.get("maps") or []

        wins = sum(int(m.get("wins", 0)) for m in maps)
        losses = sum(int(m.get("losses", 0)) for m in maps)
        games = wins + losses
        win_rate = (wins / games * 100.0) if games else 0.0

        total_g = 0
        w_kd = w_kr = w_dmg = w_score = 0.0
        for m in maps:
            g = int(m.get("wins", 0)) + int(m.get("losses", 0))
            if g <= 0:
                continue
            total_g += g
            w_kd += float(m.get("kd", 0.0)) * g
            w_kr += float(m.get("kill_round", 0.0)) * g
            w_dmg += float(m.get("dmg_round", 0.0)) * g
            w_score += float(m.get("score_round", 0.0)) * g

        summaries.append({
            "riot_id": rid,
            "games": games,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": win_rate,
            "kd": (w_kd / total_g) if total_g else 0.0,
            "kill_round": (w_kr / total_g) if total_g else 0.0,
            "dmg_round": (w_dmg / total_g) if total_g else 0.0,
            "score_round": (w_score / total_g) if total_g else 0.0,
            "profile_url": entry.get("profile_url"),
            "maps_url": entry.get("maps_url"),
            "agents_url": entry.get("agents_url"),
        })

    summaries.sort(key=lambda x: x["games"], reverse=True)

    return render_template(
        "team.html",
        team_key=team_key,
        team_name=team_name,
        seasons=snap.get("seasons") or SEASONS,
        season_id=season_id,
        team_ids=team_ids,
        summaries=summaries,
        map_rows=map_rows,
        team_summary=team_summary,
        map_plan=map_plan,
        expected_comps=expected_comps,
        team_insights=team_insights,
        snapshot_updated_ts=snap.get("updated_ts"),
    )


@app.get("/team/<team_key>/player/<riot_id>")
def player_view(team_key: str, riot_id: str):
    snap = load_snapshot(team_key)
    if not snap:
        abort(404)

    team_name = snap.get("team_name") or team_key
    team_ids = snap.get("players") or []
    if riot_id not in team_ids:
        abort(404)

    season_id = _resolve_season_id(request.args.get("seasonId") or DEFAULT_SEASON_ID, snap)

    entry = (snap.get("data") or {}).get(season_id, {}).get(riot_id) or {}

    agents = normalize_agents(entry.get("agents") or [])

    return render_template(
        "player.html",
        team_key=team_key,
        team_name=team_name,
        riot_id=riot_id,
        seasons=snap.get("seasons") or SEASONS,
        season_id=season_id,
        profile_url=entry.get("profile_url"),
        agents_url=entry.get("agents_url"),
        maps_url=entry.get("maps_url"),
        profile_stats=entry.get("profile_stats") or {},
        agents=agents,
        maps=entry.get("maps") or [],
        snapshot_updated_ts=snap.get("updated_ts"),
    )


@app.post("/team/<team_key>/update")
def team_update(team_key: str):
    snap = load_snapshot(team_key)
    if not snap:
        abort(404)

    team_name = snap.get("team_name") or team_key
    team_ids = snap.get("players") or []
    if not team_ids:
        abort(400, "No players in snapshot.")

    snap2 = build_team_snapshot(
        team_key,
        team_name,
        team_ids,
        seasons=snap.get("seasons") or SEASONS,
        polite_sleep=0.8,
    )
    save_snapshot(team_key, snap2)

    season_id = request.args.get("seasonId") or DEFAULT_SEASON_ID
    return redirect(url_for("team_view", team_key=team_key, seasonId=season_id))


@app.post("/team/<team_key>/delete")
def team_delete(team_key: str):
    p = snapshot_path(team_key)
    if p.exists():
        p.unlink()
    txt = UPLOAD_DIR / f"{team_key}.txt"
    if txt.exists():
        txt.unlink()
    return redirect(url_for("index"))


@app.get("/team/<team_key>/snapshot.json")
def download_snapshot(team_key: str):
    p = snapshot_path(team_key)
    if not p.exists():
        abort(404)
    return send_file(p, as_attachment=True)

@app.post("/dashboard/<dash_id>/delete")
def dashboard_delete(dash_id: str):
    """
    This endpoint name MUST match url_for('dashboard_delete', dash_id=...)
    """
    p = dashboard_path(dash_id)
    if p.exists():
        p.unlink()
    return redirect(url_for("dashboard"))

@app.get("/dashboard/view/<dash_id>")
def dashboard_view_imported(dash_id: str):
    dash = load_dashboard(dash_id)
    if not dash:
        abort(404)

    season_id = (request.args.get("seasonId") or "").strip() or DEFAULT_SEASON_ID

    def pick_two_snaps(d: dict):
        # 1) Preferred keys
        a = d.get("team_a_snapshot")
        b = d.get("team_b_snapshot")
        if a and b:
            return a, b

        # 2) Common alternatives
        for ka, kb in [
            ("A", "B"),
            ("snap_a", "snap_b"),
            ("snapshot_a", "snapshot_b"),
            ("teamA", "teamB"),
            ("team_a", "team_b"),
        ]:
            a = d.get(ka)
            b = d.get(kb)
            if a and b:
                return a, b

        # 3) If stored as a list/dict of teams
        teams = d.get("teams")
        if isinstance(teams, list) and len(teams) >= 2:
            if isinstance(teams[0], dict) and isinstance(teams[1], dict):
                return teams[0], teams[1]

        snaps = d.get("snapshots")
        if isinstance(snaps, dict) and len(snaps) >= 2:
            vals = list(snaps.values())
            if isinstance(vals[0], dict) and isinstance(vals[1], dict):
                return vals[0], vals[1]

        return None, None

    snap_a, snap_b = pick_two_snaps(dash)
    if not snap_a or not snap_b:
        abort(400, "Imported dashboard is missing team snapshots.")

    A = summarize_team_snapshot(snap_a, season_id)
    B = summarize_team_snapshot(snap_b, season_id)

    return render_template(
        "dashboard_imported.html",
        dash_id=dash_id,
        title=dash.get("title"),
        created_ts=dash.get("created_ts"),
        season_id=season_id,
        A=A,
        B=B,
    )

@app.post("/import")
def import_snapshot_json():
    if "file" not in request.files:
        abort(400, "Missing file")
    f = request.files["file"]
    if not f or not f.filename:
        abort(400, "Missing filename")
    if not f.filename.lower().endswith(".json"):
        abort(400, "Upload a .json snapshot")

    data = json.loads(f.read().decode("utf-8", errors="replace"))
    team_key = data.get("team_key")
    if not team_key:
        abort(400, "Invalid snapshot: missing team_key")

    save_snapshot(team_key, data)
    return redirect(url_for("team_view", team_key=team_key, seasonId=DEFAULT_SEASON_ID))


@app.post("/import-zip")
def import_snapshot_zip():
    if "file" not in request.files:
        abort(400, "Missing file")
    f = request.files["file"]
    if not f or not f.filename:
        abort(400, "Missing filename")
    if not f.filename.lower().endswith(".zip"):
        abort(400, "Upload a .zip")

    b = io.BytesIO(f.read())
    imported = 0
    last_key = None

    with zipfile.ZipFile(b, "r") as z:
        for name in z.namelist():
            if not name.lower().endswith(".json"):
                continue
            raw = z.read(name)
            try:
                data = json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                continue
            team_key = data.get("team_key")
            if not team_key:
                continue
            save_snapshot(team_key, data)
            imported += 1
            last_key = team_key

    if imported <= 0:
        abort(400, "No valid snapshot .json files found in zip.")

    return redirect(url_for("team_view", team_key=last_key, seasonId=DEFAULT_SEASON_ID))


@app.get("/export_zip")
def export_zip():
    """
    Exports ALL saved snapshots as a single zip for sharing.
    """
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in SNAPSHOT_DIR.glob("*.json"):
            z.write(p, arcname=p.name)
    mem.seek(0)
    return send_file(mem, mimetype="application/zip", as_attachment=True, download_name="snapshots.zip")


# -------------------------
# Dashboard
# -------------------------


@app.get("/dashboard")
def dashboard():
    teams = list_snapshots()
    dashboards = []
    for p in DASHBOARD_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            dashboards.append({
                "dash_id": d.get("dash_id"),
                "title": f"{d.get('team_a_name','A')} vs {d.get('team_b_name','B')}",
                "team_a_name": d.get("team_a_name"),
                "team_b_name": d.get("team_b_name"),
                "created_ts": d.get("created_ts"),
            })
        except Exception:
            continue
    dashboards.sort(key=lambda x: x.get("created_ts") or 0, reverse=True)

    return render_template(
        "dashboard.html",
        teams=teams,
        dashboards=dashboards,
        seasons=SEASONS,
        default_season=DEFAULT_SEASON_ID,
    )


@app.get("/dashboards")
def dashboards_alias():
    return redirect(url_for("dashboard"))


@app.post("/dashboard/import-zip")
def dashboard_import_zip():
    if "file" not in request.files:
        abort(400, "Missing file")
    f = request.files["file"]
    if not f or not f.filename:
        abort(400, "Missing filename")
    if not f.filename.lower().endswith(".zip"):
        abort(400, "Upload a .zip")

    b = io.BytesIO(f.read())
    snaps = []

    with zipfile.ZipFile(b, "r") as z:
        for name in z.namelist():
            if not name.lower().endswith(".json"):
                continue
            try:
                data = json.loads(z.read(name).decode("utf-8", errors="replace"))
            except Exception:
                continue
            # team snapshots should have team_key
            if isinstance(data, dict) and data.get("team_key"):
                snaps.append(data)

    if len(snaps) < 2:
        abort(400, "Zip must contain at least TWO team snapshot .json files.")

    # pick first two (or add logic to choose specific ones)
    snap_a, snap_b = snaps[0], snaps[1]

    dash_id = uuid.uuid4().hex[:10]
    title = f"{snap_a.get('team_name') or snap_a.get('team_key')} vs {snap_b.get('team_name') or snap_b.get('team_key')}"

    dash = {
        "title": title,
        "created_ts": int(time.time()),
        "team_a_snapshot": snap_a,
        "team_b_snapshot": snap_b,
    }
    save_dashboard(dash_id, dash)

    return redirect(url_for("dashboard_view_imported", dash_id=dash_id))

    if not dash or not dash.get("dash_id"):
        abort(400, "dashboard.json missing from zip (or invalid).")

    (DASHBOARD_DIR / f"{dash['dash_id']}.json").write_text(json.dumps(dash, indent=2), encoding="utf-8")
    return redirect(url_for("dashboard_view_imported", dash_id=dash["dash_id"]))


# -------------------------
# Compare
# -------------------------

@app.get("/compare")
def compare_picker():
    teams = list_snapshots()
    return render_template(
        "compare_picker.html",
        teams=teams,
        seasons=SEASONS,
        default_season=DEFAULT_SEASON_ID,
    )


@app.get("/compare/run")
def compare_run():
    """
    /compare/run?a=<team_key>&b=<team_key>&seasonId=<season_id>
    """
    a = (request.args.get("a") or "").strip()
    b = (request.args.get("b") or "").strip()
    season_id = request.args.get("seasonId") or DEFAULT_SEASON_ID

    if not a or not b or a == b:
        abort(400, "Pick two different teams to compare.")

    snap_a = load_snapshot(a)
    snap_b = load_snapshot(b)
    if not snap_a or not snap_b:
        abort(404, "One or both snapshots not found.")

    if season_id not in (snap_a.get("data") or {}):
        season_id = DEFAULT_SEASON_ID
    if season_id not in (snap_b.get("data") or {}):
        season_id = DEFAULT_SEASON_ID

    A = summarize_team_snapshot(snap_a, season_id)
    B = summarize_team_snapshot(snap_b, season_id)
    insights = generate_comparison_insights(A, B)
    dash = save_comparison_dashboard(snap_a, snap_b, season_id, A, B, insights)

    return render_template(
        "compare_view.html",
        season_id=season_id,
        seasons=SEASONS,
        team_a=A,
        team_b=B,
        insights=insights,
        compare_id=dash["dash_id"],
    )


@app.get("/compare/export/<compare_id>")
def export_comparison(compare_id: str):
    dash_path = DASHBOARD_DIR / f"{compare_id}.json"
    if not dash_path.exists():
        abort(404)

    dash = json.loads(dash_path.read_text(encoding="utf-8"))
    a_key = dash.get("team_a_key")
    b_key = dash.get("team_b_key")

    a_path = snapshot_path(a_key) if a_key else None
    b_path = snapshot_path(b_key) if b_key else None
    if not a_path or not a_path.exists() or not b_path or not b_path.exists():
        abort(400, "Missing one or both team snapshots required for export.")

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("dashboard.json", dash_path.read_text(encoding="utf-8"))
        z.writestr(f"teamA_{a_key}.json", a_path.read_text(encoding="utf-8"))
        z.writestr(f"teamB_{b_key}.json", b_path.read_text(encoding="utf-8"))

    mem.seek(0)
    return send_file(mem, mimetype="application/zip", as_attachment=True, download_name=f"comparison_{compare_id}.zip")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
