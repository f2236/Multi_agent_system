"""Streamlit dashboard for real-time Groq-planned UAV optimization.

The dashboard intentionally avoids Plotly so a pandas/Plotly import-state issue
cannot crash rendering after a successful Groq response.
"""

from copy import deepcopy
from html import escape
import json
import math
import time


COLORS = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2"]


def run_dashboard(orchestrator, uavs, antennas, iterations=8, delay=0.5):
    try:
        import streamlit as st
    except Exception as e:
        raise RuntimeError("Streamlit is required to run the dashboard") from e

    st.set_page_config(page_title="UAV Groq Optimization", layout="wide")
    _inject_styles(st)
    _initialize_state(st, orchestrator, uavs, antennas)

    st.markdown(
        """
        <div class="page-heading">
          <div>Llama 3.3 Multi-Agent UAV Trajectory Optimization</div>
          <span>Groq plans movement. Local evaluator enforces constraints and sum-rate proof.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_uavs = st.session_state.scenario_uavs
    active_antennas = st.session_state.scenario_antennas

    st.sidebar.header("Controls")
    groq_key = st.sidebar.text_input("Groq API key", type="password")
    groq_model = st.sidebar.text_input(
        "Groq model",
        value=getattr(orchestrator.planner, "model", "llama-3.3-70b-versatile"),
    )
    if groq_key and hasattr(orchestrator.planner, "api_key"):
        orchestrator.planner.api_key = groq_key
    if groq_model and hasattr(orchestrator.planner, "model"):
        orchestrator.planner.model = groq_model

    max_iters = st.sidebar.slider("Iterations per run", 1, 20, value=min(iterations, 8))
    delay = st.sidebar.slider("Delay between iterations (s)", 0.0, 2.0, value=delay, step=0.1)
    run_one = st.sidebar.button("Run 1 Groq Iteration")
    run_batch = st.sidebar.button("Run Live Optimization")
    show_proof = st.sidebar.button("Show Optimization Proof")
    reset = st.sidebar.button("Reset Scenario")

    if reset:
        _reset_state(st, orchestrator, uavs, antennas)
        active_uavs = st.session_state.scenario_uavs
        active_antennas = st.session_state.scenario_antennas

    if show_proof:
        st.session_state.show_proof = True

    main_slot = st.empty()

    def render(error=None):
        st.session_state.render_seq += 1
        render_seq = st.session_state.render_seq
        with main_slot.container():
            _render_summary(st, orchestrator)
            st.markdown(
                _simulation_svg(
                    active_uavs,
                    active_antennas,
                    st.session_state.initial_positions,
                    st.session_state.trajectory_history,
                    orchestrator.config,
                ),
                unsafe_allow_html=True,
            )

            if error:
                st.error(error)

            st.caption("Proof, Groq prompts, and audit logs are kept below so the live simulation stays clean.")

            with st.expander("Optimization proof and objective history", expanded=st.session_state.show_proof):
                proof_left, proof_right = st.columns([1.0, 1.2])
                with proof_left:
                    st.markdown(_proof_cards(st), unsafe_allow_html=True)
                    st.download_button(
                        "Download Proof Report",
                        _proof_text(st),
                        file_name="uav_optimization_proof.txt",
                        mime="text/plain",
                        key=f"proof_download_live_{render_seq}",
                    )
                with proof_right:
                    st.markdown(_rate_history_svg(st.session_state.history), unsafe_allow_html=True)

            with st.expander("Iteration audit table", expanded=False):
                st.markdown(_records_table(st.session_state.records[-12:]), unsafe_allow_html=True)

            with st.expander("Raw live logs", expanded=False):
                st.code("\n".join(st.session_state.planner_logs[-240:]) or "No iterations run yet.")

    if run_one or run_batch:
        count = 1 if run_one else max_iters
        for _ in range(count):
            iteration = len(st.session_state.records) + 1
            try:
                before_requests = getattr(orchestrator.planner, "request_count", 0)
                result = orchestrator.run_iteration(active_uavs, active_antennas, iteration=iteration)
                after_requests = getattr(orchestrator.planner, "request_count", before_requests)
            except Exception as e:
                logs = getattr(orchestrator, "last_iteration_logs", None) or [
                    f"Iteration: {iteration}",
                    f"Planner error: {e}",
                ]
                st.session_state.planner_logs.extend(logs)
                render(error=f"Groq planning failed: {e}")
                break

            _record_iteration(st, result, after_requests - before_requests)
            render()
            if delay:
                time.sleep(delay)
    else:
        render()

    st.sidebar.write("Groq key configured:", "yes" if getattr(orchestrator.planner, "api_key", None) else "no")
    st.sidebar.write("Groq requests in this run:", st.session_state.groq_requests)


def _initialize_state(st, orchestrator, uavs, antennas):
    if "scenario_uavs" not in st.session_state:
        _reset_state(st, orchestrator, uavs, antennas)
    if "show_proof" not in st.session_state:
        st.session_state.show_proof = False
    if "render_seq" not in st.session_state:
        st.session_state.render_seq = 0


def _reset_state(st, orchestrator, uavs, antennas):
    st.session_state.scenario_uavs = deepcopy(uavs)
    st.session_state.scenario_antennas = deepcopy(antennas)
    st.session_state.initial_positions = [tuple(u.pos) for u in st.session_state.scenario_uavs]
    initial_rate = orchestrator.evaluator.evaluate(
        st.session_state.scenario_uavs,
        st.session_state.scenario_antennas,
    )
    st.session_state.initial_rate = initial_rate
    st.session_state.current_rate = initial_rate
    st.session_state.best_rate = initial_rate
    st.session_state.best_iteration = 0
    st.session_state.history = [initial_rate]
    st.session_state.records = []
    st.session_state.planner_logs = []
    st.session_state.groq_requests = 0
    st.session_state.trajectory_history = [list(st.session_state.initial_positions)]


def _record_iteration(st, result, request_delta):
    current_positions = [tuple(pos) for pos in result["current_positions"]]
    old_rate = float(result["old_rate"])
    new_rate = float(result["new_rate"])
    score = float(result["score"])

    st.session_state.current_rate = score
    st.session_state.history.append(score)
    st.session_state.trajectory_history.append(current_positions)
    st.session_state.groq_requests += max(0, int(request_delta))
    if score >= st.session_state.best_rate:
        st.session_state.best_rate = score
        st.session_state.best_iteration = result["iteration"]

    st.session_state.records.append(
        {
            "iteration": result["iteration"],
            "movement": result["directions"],
            "accepted": result["accepted"],
            "decision": result.get("decision", "Accepted" if result["accepted"] else "Rejected"),
            "selected_candidate_id": result.get("selected_candidate_id", ""),
            "candidate_moves": result.get("candidate_moves", []),
            "old_rate": old_rate,
            "new_rate": new_rate,
            "score": score,
            "old_positions": result["old_positions"],
            "suggested_positions": result["suggested_positions"],
            "current_positions": current_positions,
            "prompt": result.get("prompt", ""),
            "response": result.get("response_text", ""),
            "planning_source": result.get("planning_source", "Groq Llama 3"),
            "planner_error": result.get("planner_error", ""),
        }
    )
    st.session_state.planner_logs.extend(result["logs"])


def _render_summary(st, orchestrator):
    initial = st.session_state.initial_rate
    current = st.session_state.current_rate
    best = st.session_state.best_rate
    improvement = current - initial
    improvement_pct = (improvement / initial * 100.0) if initial else 0.0
    accepted = sum(1 for r in st.session_state.records if r.get("decision") == "Accepted")
    noop = sum(1 for r in st.session_state.records if r.get("decision") == "No-op")
    rejected = len(st.session_state.records) - accepted - noop
    last_decision = st.session_state.records[-1].get("decision", "Ready") if st.session_state.records else "Ready"
    last_source = st.session_state.records[-1].get("planning_source", "Ready") if st.session_state.records else "Ready"

    st.markdown(
        f"""
        <div class="mission-strip">
          {_metric_card("Initial", _fmt_rate(initial), "before planning")}
          {_metric_card("Current", _fmt_rate(current), f"{improvement_pct:+.2f}%")}
          {_metric_card("Best", _fmt_rate(best), f"iteration {st.session_state.best_iteration}")}
          {_metric_card("Groq Calls", str(st.session_state.groq_requests), "one per iteration")}
          {_metric_card("Decision", escape(last_decision), f"A:{accepted} N:{noop} R:{rejected}")}
          {_metric_card("Planner", escape(last_source), escape(getattr(orchestrator.planner, "model", "Groq")))}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_proof_report(st):
    initial = st.session_state.initial_rate
    current = st.session_state.current_rate
    best = st.session_state.best_rate
    improvement = current - initial
    accepted = sum(1 for r in st.session_state.records if r.get("decision") == "Accepted")
    expected = len(st.session_state.records)
    actual = st.session_state.groq_requests
    fallback_count = sum(1 for r in st.session_state.records if "fallback" in r.get("planning_source", "").lower())
    last = st.session_state.records[-1] if st.session_state.records else None

    st.subheader("Optimization Proof Report")
    st.markdown(
        f"""
        <div class="proof-box">
          <b>Initial objective:</b> {_fmt_rate(initial)}<br>
          <b>Current objective:</b> {_fmt_rate(current)}<br>
          <b>Best objective:</b> {_fmt_rate(best)}<br>
          <b>Net objective change:</b> {_fmt_rate(improvement)}<br>
          <b>Antenna-port setup:</b> K = 1, fixed FAS port 1; no antenna/port selection is optimized.<br>
          <b>Groq request proof:</b> {actual} requests recorded for {expected} optimization iterations.<br>
          <b>Continuity fallback:</b> {fallback_count} iterations used the local best-candidate safety planner.<br>
          <b>Acceptance proof:</b> {accepted} accepted movements were applied only when local sum-rate did not decrease.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if last:
        with st.expander("Last prompt sent to Llama 3.3"):
            st.code(last["prompt"] or "No prompt recorded.")
        with st.expander("Last Llama 3.3 response"):
            st.code(last["response"] or "No response recorded.")

    report_text = _proof_text(st)
    st.download_button(
        "Download Proof Report",
        report_text,
        file_name="uav_optimization_proof.txt",
        mime="text/plain",
        key=f"proof_download_report_{st.session_state.render_seq}",
    )


def _proof_cards(st):
    initial = st.session_state.initial_rate
    current = st.session_state.current_rate
    best = st.session_state.best_rate
    delta = current - initial
    status = "Optimized" if delta >= 0 else "Not improved yet"
    return f"""
    <div class="proof-box">
      <b>Proof status:</b> {status}<br>
      <b>Before:</b> {_fmt_rate(initial)}<br>
      <b>After:</b> {_fmt_rate(current)}<br>
      <b>Best observed:</b> {_fmt_rate(best)}<br>
      <b>Antenna-port setup:</b> K = 1, fixed FAS port 1.<br>
      <b>Why this is proof:</b> every accepted move is evaluated by the local sum-rate formula after the planner proposes movement.
    </div>
    """


def _records_table(records):
    if not records:
        return "<div class='empty'>No optimization iterations yet.</div>"

    rows = []
    for record in records:
        status = record.get("decision") or ("Accepted" if record["accepted"] else "Rejected")
        selected = record.get("selected_candidate_id") or "custom"
        candidate_delta = _selected_candidate_delta(record)
        rows.append(
            "<tr>"
            f"<td>{record['iteration']}</td>"
            f"<td>{escape(record.get('planning_source', ''))}</td>"
            f"<td>{escape(selected)}</td>"
            f"<td>{_fmt_rate(candidate_delta)}</td>"
            f"<td>{escape(str(record['movement']))}</td>"
            f"<td>{_fmt_rate(record['old_rate'])}</td>"
            f"<td>{_fmt_rate(record['new_rate'])}</td>"
            f"<td>{_fmt_rate(record['score'])}</td>"
            f"<td><span class='badge {_status_class(status)}'>{status}</span></td>"
            "</tr>"
        )
    return (
        "<table class='audit-table'>"
        "<thead><tr><th>Iteration</th><th>Planning source</th><th>Candidate</th><th>Candidate delta</th><th>Movement</th><th>Old rate</th>"
        "<th>Suggested rate</th><th>Applied rate</th><th>Decision</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _selected_candidate_delta(record):
    selected = record.get("selected_candidate_id")
    for candidate in record.get("candidate_moves", []):
        if candidate.get("candidate_id") == selected:
            return float(candidate.get("delta_rate_bps", 0.0))
    return float(record["new_rate"]) - float(record["old_rate"])


def _status_class(status):
    normalized = str(status).lower()
    if "reject" in normalized:
        return "rejected"
    if "no-op" in normalized:
        return "noop"
    return "accepted"


def _simulation_svg(uavs, antennas, initial_positions, trajectory_history, config):
    width, height = 1180, 620
    x_min, x_max = config.grid_x_bounds
    y_min, y_max = config.grid_y_bounds
    z_min, z_max = float(config.min_altitude), float(config.max_altitude)
    earth_cx, earth_cy, earth_r = width / 2, height + 96, 348

    def clamp(value, low=0.0, high=1.0):
        return max(low, min(high, float(value)))

    def norm(value, low, high):
        return clamp((float(value) - float(low)) / max(1e-9, float(high) - float(low)))

    def screen_from_pos(pos):
        x, y, z = _position_tuple(pos)
        x_ratio = norm(x, x_min, x_max)
        y_ratio = norm(y, y_min, y_max)
        z_ratio = norm(z, z_min, z_max)
        angle = math.radians(218 + x_ratio * 124)
        orbit_radius = earth_r + 112 + z_ratio * 96 + (y_ratio - 0.5) * 52
        sx = earth_cx + math.cos(angle) * orbit_radius
        sy = earth_cy + math.sin(angle) * orbit_radius
        return sx, sy

    def ground_from_pos(pos):
        x, y, _ = _position_tuple(pos)
        x_ratio = norm(x, x_min, x_max)
        y_ratio = norm(y, y_min, y_max)
        angle = math.radians(238 + x_ratio * 64)
        surface = earth_r + 6 + (y_ratio - 0.5) * 18
        gx = earth_cx + math.cos(angle) * surface
        gy = earth_cy + math.sin(angle) * surface
        return gx, gy

    stars = []
    for i in range(96):
        sx = (i * 97 + 31) % width
        sy = (i * 53 + 19) % 410
        radius = 0.7 + (i % 3) * 0.35
        opacity = 0.45 + (i % 5) * 0.1
        delay = (i % 8) * 0.25
        stars.append(
            f"<circle cx='{sx:.1f}' cy='{sy:.1f}' r='{radius:.1f}' opacity='{opacity:.2f}' "
            f"class='star' style='animation-delay:{delay:.2f}s'/>"
        )

    antenna_points = []
    for idx, antenna in enumerate(antennas):
        pos = antenna.get("pos") if isinstance(antenna, dict) else getattr(antenna, "pos", None)
        label = antenna.get("label", "FAS port 1 (K=1)") if isinstance(antenna, dict) else "FAS port 1 (K=1)"
        antenna_points.append((idx, label, *ground_from_pos(pos)))

    elements = [
        "<div class='space-shell'>",
        f"<svg viewBox='0 0 {width} {height}' class='space-svg' role='img'>",
        "<defs>",
        "<radialGradient id='spaceGlow' cx='50%' cy='38%' r='70%'>"
        "<stop offset='0%' stop-color='#1e3a8a' stop-opacity='0.45'/>"
        "<stop offset='48%' stop-color='#0f172a' stop-opacity='0.82'/>"
        "<stop offset='100%' stop-color='#020617' stop-opacity='1'/>"
        "</radialGradient>",
        "<radialGradient id='earthOcean' cx='44%' cy='32%' r='68%'>"
        "<stop offset='0%' stop-color='#67e8f9'/>"
        "<stop offset='42%' stop-color='#0ea5e9'/>"
        "<stop offset='100%' stop-color='#0f3d91'/>"
        "</radialGradient>",
        "<filter id='softGlow'><feGaussianBlur stdDeviation='4' result='blur'/><feMerge><feMergeNode in='blur'/><feMergeNode in='SourceGraphic'/></feMerge></filter>",
        "</defs>",
        "<rect width='1180' height='620' fill='url(#spaceGlow)'/>",
        *stars,
        "<circle cx='930' cy='115' r='58' fill='#f8fafc' opacity='0.08'/>",
        "<circle cx='930' cy='115' r='37' fill='#dbeafe' opacity='0.12'/>",
        "<text x='38' y='48' class='space-title'>Live Llama UAV Mission</text>",
        "<text x='40' y='76' class='space-subtitle'>Groq plans movement, local evaluator accepts only non-decreasing sum-rate moves.</text>",
        f"<text x='40' y='104' class='space-subtitle'>Movement step: {float(config.uav_speed * config.time_step):.1f} m | K fixed to 1</text>",
        f"<ellipse cx='{earth_cx:.1f}' cy='{earth_cy:.1f}' rx='{earth_r + 156:.1f}' ry='{earth_r * 0.53:.1f}' class='orbit-ring orbit-a'/>",
        f"<ellipse cx='{earth_cx:.1f}' cy='{earth_cy:.1f}' rx='{earth_r + 76:.1f}' ry='{earth_r * 0.42:.1f}' class='orbit-ring orbit-b'/>",
    ]
    elements.extend(
        [
            f"<circle cx='{earth_cx:.1f}' cy='{earth_cy:.1f}' r='{earth_r + 18:.1f}' class='earth-atmosphere'/>",
            f"<g class='earth-spin' style='transform-origin:{earth_cx:.1f}px {earth_cy:.1f}px'>",
            f"<circle cx='{earth_cx:.1f}' cy='{earth_cy:.1f}' r='{earth_r:.1f}' fill='url(#earthOcean)'/>",
            f"<path d='M {earth_cx - 230:.1f} {earth_cy - 240:.1f} C {earth_cx - 150:.1f} {earth_cy - 300:.1f}, {earth_cx - 58:.1f} {earth_cy - 252:.1f}, {earth_cx - 88:.1f} {earth_cy - 185:.1f} C {earth_cx - 128:.1f} {earth_cy - 98:.1f}, {earth_cx - 10:.1f} {earth_cy - 92:.1f}, {earth_cx + 52:.1f} {earth_cy - 146:.1f} C {earth_cx + 118:.1f} {earth_cy - 204:.1f}, {earth_cx + 198:.1f} {earth_cy - 176:.1f}, {earth_cx + 226:.1f} {earth_cy - 116:.1f}' class='land-mass'/>",
            f"<path d='M {earth_cx - 110:.1f} {earth_cy - 54:.1f} C {earth_cx - 18:.1f} {earth_cy - 112:.1f}, {earth_cx + 68:.1f} {earth_cy - 80:.1f}, {earth_cx + 142:.1f} {earth_cy - 22:.1f} C {earth_cx + 60:.1f} {earth_cy + 18:.1f}, {earth_cx - 26:.1f} {earth_cy + 12:.1f}, {earth_cx - 110:.1f} {earth_cy - 54:.1f}' class='land-mass secondary'/>",
            f"<path d='M {earth_cx - 286:.1f} {earth_cy - 72:.1f} C {earth_cx - 184:.1f} {earth_cy - 126:.1f}, {earth_cx - 84:.1f} {earth_cy - 132:.1f}, {earth_cx + 18:.1f} {earth_cy - 86:.1f}' class='cloud-band'/>",
            f"<path d='M {earth_cx + 20:.1f} {earth_cy - 268:.1f} C {earth_cx + 102:.1f} {earth_cy - 302:.1f}, {earth_cx + 198:.1f} {earth_cy - 270:.1f}, {earth_cx + 268:.1f} {earth_cy - 216:.1f}' class='cloud-band thin'/>",
            "</g>",
            f"<circle cx='{earth_cx:.1f}' cy='{earth_cy:.1f}' r='{earth_r:.1f}' class='earth-shade'/>",
        ]
    )

    for idx, frame in enumerate(trajectory_history[-7:]):
        if not frame:
            continue
        opacity = 0.08 + idx * 0.025
        frame_points = []
        for pos in frame:
            px, py = screen_from_pos(pos)
            frame_points.append(f"{px:.1f},{py:.1f}")
        elements.append(
            f"<polyline points='{' '.join(frame_points)}' "
            f"fill='none' stroke='#93c5fd' stroke-width='1.2' stroke-opacity='{opacity:.2f}'/>"
        )

    for idx, label, antenna_x, antenna_y in antenna_points:
        elements.append(f"<circle cx='{antenna_x:.1f}' cy='{antenna_y:.1f}' r='12' class='ground-station'/>")
        elements.append(f"<path d='M {antenna_x - 14:.1f} {antenna_y + 16:.1f} L {antenna_x:.1f} {antenna_y - 8:.1f} L {antenna_x + 14:.1f} {antenna_y + 16:.1f}' class='station-mast'/>")
        elements.append(f"<text x='{antenna_x + 20:.1f}' y='{antenna_y + 5:.1f}' class='ground-label'>{escape(label)}</text>")

    for idx, uav in enumerate(uavs):
        color = COLORS[idx % len(COLORS)]
        pos = getattr(uav, "pos", uav)
        x, y, z = _position_tuple(pos)
        ux, uy = screen_from_pos(pos)
        ix, iy = screen_from_pos(initial_positions[idx])

        path_points = []
        for frame in trajectory_history:
            if idx < len(frame):
                px, py = screen_from_pos(frame[idx])
                path_points.append(f"{px:.1f},{py:.1f}")
        if len(path_points) > 1:
            elements.append(
                f"<polyline points='{' '.join(path_points)}' fill='none' stroke='{color}' "
                "stroke-width='3' stroke-opacity='0.55' class='uav-trail'/>"
            )

        nearest = min(
            antenna_points,
            key=lambda item: (item[2] - ux) ** 2 + (item[3] - uy) ** 2,
        ) if antenna_points else None
        if nearest:
            _, _label, ax, ay = nearest
            elements.append(f"<line x1='{ux:.1f}' y1='{uy:.1f}' x2='{ax:.1f}' y2='{ay:.1f}' class='signal-beam'/>")

        elements.append(f"<circle cx='{ix:.1f}' cy='{iy:.1f}' r='8' fill='none' stroke='{color}' stroke-dasharray='4 4' opacity='0.7'/>")
        elements.append(f"<circle cx='{ux:.1f}' cy='{uy:.1f}' r='20' fill='none' stroke='{color}' stroke-opacity='0.35' class='uav-pulse'/>")
        elements.append(f"<g class='satellite' filter='url(#softGlow)'>")
        elements.append(f"<rect x='{ux - 11:.1f}' y='{uy - 6:.1f}' width='22' height='12' rx='3' fill='{color}'/>")
        elements.append(f"<rect x='{ux - 30:.1f}' y='{uy - 4:.1f}' width='14' height='8' rx='2' fill='#bae6fd' opacity='0.9'/>")
        elements.append(f"<rect x='{ux + 16:.1f}' y='{uy - 4:.1f}' width='14' height='8' rx='2' fill='#bae6fd' opacity='0.9'/>")
        elements.append(f"<circle cx='{ux:.1f}' cy='{uy:.1f}' r='3.5' fill='#f8fafc'/>")
        elements.append("</g>")
        elements.append(
            f"<text x='{ux + 24:.1f}' y='{uy - 8:.1f}' class='uav-label'>UAV {getattr(uav, 'uav_id', idx)}</text>"
        )
        elements.append(
            f"<text x='{ux + 24:.1f}' y='{uy + 12:.1f}' class='uav-sub-label'>({x:.0f}, {y:.0f}, {z:.0f}) m</text>"
        )

    elements.extend(["</svg>", "</div>"])
    return "".join(elements)


def _rate_history_svg(history):
    width, height = 520, 230
    if len(history) < 2:
        return "<div class='empty'>Run iterations to build objective history.</div>"

    values = [float(v) for v in history]
    lo, hi = min(values), max(values)
    span = max(1e-9, hi - lo)
    points = []
    for idx, value in enumerate(values):
        x = 28 + idx / max(1, len(values) - 1) * (width - 56)
        y = height - 34 - (value - lo) / span * (height - 72)
        points.append(f"{x:.1f},{y:.1f}")

    return (
        f"<svg viewBox='0 0 {width} {height}' class='history-svg'>"
        "<rect x='8' y='8' width='504' height='214' rx='8' fill='#f8fafc' stroke='#cbd5e1'/>"
        "<text x='22' y='30' class='svg-title'>Stable sum data rate over iterations</text>"
        f"<polyline points='{' '.join(points)}' fill='none' stroke='#2563eb' stroke-width='4'/>"
        f"<circle cx='{points[-1].split(',')[0]}' cy='{points[-1].split(',')[1]}' r='6' fill='#2563eb'/>"
        f"<text x='22' y='{height - 12}' class='label'>min {_fmt_rate(lo)} | max {_fmt_rate(hi)}</text>"
        "</svg>"
    )


def _proof_text(st):
    records = st.session_state.records
    lines = [
        "UAV Llama 3.3 Optimization Proof",
        f"Initial rate: {_fmt_rate(st.session_state.initial_rate)}",
        f"Current rate: {_fmt_rate(st.session_state.current_rate)}",
        f"Best rate: {_fmt_rate(st.session_state.best_rate)}",
        f"Groq requests: {st.session_state.groq_requests}",
        "Planner policy: Groq Llama 3 first; local best-candidate continuity fallback if Groq is unavailable or invalid.",
        "Antenna-port setup: K=1, fixed FAS port 1; no antenna/port selection optimization.",
        "",
        "Iterations:",
    ]
    for r in records:
        lines.append(
            f"{r['iteration']}: source={r.get('planning_source', '')} "
            f"candidate={r.get('selected_candidate_id') or 'custom'} "
            f"candidate_delta={_fmt_rate(_selected_candidate_delta(r))} movement={r['movement']} "
            f"old={_fmt_rate(r['old_rate'])} new={_fmt_rate(r['new_rate'])} "
            f"applied={_fmt_rate(r['score'])} decision={r.get('decision')}"
        )
    return "\n".join(lines)


def _metric_card(title, value, detail):
    return (
        "<div class='metric-card'>"
        f"<div class='metric-title'>{title}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"<div class='metric-detail'>{detail}</div>"
        "</div>"
    )


def _fmt_rate(value):
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e9:
        return f"{sign}{value / 1e9:.3f} Gbps"
    if value >= 1e6:
        return f"{sign}{value / 1e6:.3f} Mbps"
    if value >= 1e3:
        return f"{sign}{value / 1e3:.3f} Kbps"
    return f"{sign}{value:.3f} bps"


def _position_tuple(pos):
    if hasattr(pos, "to_tuple"):
        pos = pos.to_tuple()
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def _inject_styles(st):
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            max-width: 1440px;
        }
        .page-heading {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 16px;
            margin: 0 0 8px;
        }
        .page-heading div {
            font-size: 22px;
            font-weight: 800;
            color: #0f172a;
        }
        .page-heading span {
            font-size: 13px;
            color: #475569;
        }
        .mission-strip {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 8px;
            margin: 8px 0 10px;
        }
        .metric-card {
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 8px;
            padding: 9px 10px;
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            box-shadow: 0 8px 26px rgba(15, 23, 42, 0.12);
        }
        .metric-title {
            font-size: 11px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .metric-value {
            font-size: 18px;
            font-weight: 700;
            color: #f8fafc;
            margin-top: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .metric-detail {
            font-size: 12px;
            color: #cbd5e1;
            margin-top: 2px;
        }
        .proof-box, .empty {
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 12px;
            background: #f8fafc;
            color: #111827;
            margin-bottom: 12px;
        }
        .audit-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-bottom: 12px;
        }
        .audit-table th, .audit-table td {
            border: 1px solid #d1d5db;
            padding: 7px 8px;
            text-align: left;
            vertical-align: top;
        }
        .audit-table th {
            background: #f1f5f9;
            color: #334155;
        }
        .badge {
            border-radius: 999px;
            padding: 3px 8px;
            font-size: 12px;
            font-weight: 700;
        }
        .accepted {
            background: #dcfce7;
            color: #166534;
        }
        .rejected {
            background: #fee2e2;
            color: #991b1b;
        }
        .noop {
            background: #e0f2fe;
            color: #075985;
        }
        .sim-svg, .history-svg {
            width: 100%;
            height: auto;
            display: block;
        }
        .space-shell {
            overflow: hidden;
            border-radius: 8px;
            border: 1px solid rgba(125, 211, 252, 0.22);
            background: #020617;
            box-shadow: 0 24px 70px rgba(2, 6, 23, 0.28);
        }
        .space-svg {
            width: 100%;
            min-height: 520px;
            display: block;
        }
        .space-title {
            font-size: 28px;
            font-weight: 800;
            fill: #f8fafc;
        }
        .space-subtitle {
            font-size: 14px;
            fill: #bae6fd;
            opacity: 0.9;
        }
        .star {
            fill: #ffffff;
            animation: twinkle 3.2s ease-in-out infinite;
        }
        .orbit-ring {
            fill: none;
            stroke: #38bdf8;
            stroke-width: 1.2;
            stroke-dasharray: 7 12;
            opacity: 0.26;
        }
        .orbit-a {
            animation: orbit-dash 16s linear infinite;
        }
        .orbit-b {
            animation: orbit-dash 22s linear infinite reverse;
        }
        .signal-beam {
            stroke: #67e8f9;
            stroke-width: 1.4;
            stroke-dasharray: 8 10;
            opacity: 0.62;
            animation: orbit-dash 2.6s linear infinite;
        }
        .uav-trail {
            stroke-linecap: round;
            stroke-linejoin: round;
        }
        .uav-pulse {
            animation: pulse 1.8s ease-out infinite;
        }
        .satellite rect {
            stroke: rgba(255, 255, 255, 0.42);
            stroke-width: 0.7;
        }
        .uav-label {
            font-size: 12px;
            fill: #f8fafc;
            font-weight: 700;
        }
        .uav-sub-label {
            font-size: 11px;
            fill: #bfdbfe;
        }
        .ground-station {
            fill: #f8fafc;
            stroke: #38bdf8;
            stroke-width: 3;
        }
        .station-mast {
            fill: none;
            stroke: #f8fafc;
            stroke-width: 3;
            stroke-linecap: round;
            stroke-linejoin: round;
        }
        .ground-label {
            font-size: 12px;
            fill: #e0f2fe;
            font-weight: 700;
        }
        .earth-atmosphere {
            fill: rgba(125, 211, 252, 0.12);
            stroke: rgba(186, 230, 253, 0.28);
            stroke-width: 2;
        }
        .earth-spin {
            animation: earth-spin 34s linear infinite;
        }
        .land-mass {
            fill: #22c55e;
            opacity: 0.74;
        }
        .land-mass.secondary {
            fill: #16a34a;
            opacity: 0.64;
        }
        .cloud-band {
            fill: none;
            stroke: rgba(255, 255, 255, 0.74);
            stroke-width: 14;
            stroke-linecap: round;
            opacity: 0.78;
        }
        .cloud-band.thin {
            stroke-width: 8;
            opacity: 0.56;
        }
        .earth-shade {
            fill: rgba(2, 6, 23, 0.18);
            stroke: rgba(255, 255, 255, 0.18);
            stroke-width: 1;
        }
        @keyframes twinkle {
            0%, 100% { opacity: 0.34; }
            50% { opacity: 1; }
        }
        @keyframes pulse {
            0% { stroke-width: 3; opacity: 0.58; }
            70% { stroke-width: 1; opacity: 0.08; }
            100% { stroke-width: 1; opacity: 0; }
        }
        @keyframes orbit-dash {
            to { stroke-dashoffset: -120; }
        }
        @keyframes earth-spin {
            to { transform: rotate(360deg); }
        }
        @media (max-width: 900px) {
            .page-heading {
                display: block;
            }
            .page-heading span {
                display: block;
                margin-top: 4px;
            }
            .mission-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .space-svg {
                min-height: 430px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
