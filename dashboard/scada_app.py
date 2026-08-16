import streamlit as st
import plotly.graph_objects as go
import requests
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime

from config.settings import MACHINES, FLASK_URL
from training.feature_engineering import engineer_features

st.set_page_config(page_title="SCADA Monitor", layout="wide")

CSS = """
<style>
    .stApp {background-color: #0a0e17;}
    .main-header {background: linear-gradient(90deg, #0d1b2a, #1b263b, #0d1b2a);
        padding: 10px 20px; border-bottom: 3px solid #00e676; margin-bottom: 20px;
        text-align: center; font-size: 26px; font-weight: bold; color: #e0e0e0;
        font-family: monospace; letter-spacing: 3px;}
    .alarm-bar {background: #ff1744; color: white; padding: 8px 15px; border-radius: 4px;
        font-family: monospace; font-weight: bold; margin: 3px 0;}
    .warn-bar {background: #ffc107; color: black; padding: 8px 15px; border-radius: 4px;
        font-family: monospace; font-weight: bold; margin: 3px 0;}
    .ok-bar {background: #0d47a1; color: white; padding: 6px 15px; border-radius: 4px;
        font-family: monospace; margin: 3px 0;}
    .nameplate {background: #0a0e17; border: 2px solid #455a64; border-radius: 6px;
        padding: 12px; font-family: monospace; color: #b0bec5; font-size: 13px;}
    div[data-testid="stMetric"] {background: #0d1b2a; border: 1px solid #1e3a5f;
        border-radius: 6px; padding: 8px;}
    div[data-testid="stMetric"] label {color: #64b5f6 !important; font-family: monospace !important;}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {color: #e0e0e0 !important;}
    section[data-testid="stSidebar"] {background: #0a0e17; border-right: 1px solid #1e3a5f;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_data
def load_and_precompute():
    """Load each machine's CSV and engineer features via the same function
    training/train_models.py uses, so the dashboard can't drift from what
    the model was trained on."""
    data = {}
    missing = []
    for mid, info in MACHINES.items():
        if not os.path.exists(info["csv"]):
            missing.append(info["csv"])
            continue
        df = pd.read_csv(info["csv"])
        df["machine"] = mid
        data[mid] = engineer_features(df)
    if missing:
        st.warning("No training data yet for: %s. Run `python -m simulator.generate_data` "
                   "first (see SETUP.md)." % ", ".join(missing))
    return data

@st.cache_data
def get_indices():
    csv = load_and_precompute()
    idx = {}
    for mid, df in csv.items():
        idx[mid] = {
            "normal": df[df["machine_state"]=="normal"].index.tolist()[:500],
            "early": df[df["machine_state"]=="early_fault"].index.tolist()[:500],
            "critical": df[df["machine_state"]=="critical_fault"].index.tolist()[:500],
            "sensor": df[df["sensor_health"]!="all_ok"].index.tolist()[:500]
        }
    return idx

csv_data = load_and_precompute()
fault_idx = get_indices()

for key in ["readings","alerts","prev","page","detail","row","scenario"]:
    if key not in st.session_state:
        if key == "readings": st.session_state[key] = {m: [] for m in MACHINES}
        elif key == "alerts": st.session_state[key] = []
        elif key == "prev": st.session_state[key] = {}
        elif key == "page": st.session_state[key] = "overview"
        elif key == "detail": st.session_state[key] = "motor"
        elif key == "row": st.session_state[key] = 0
        elif key == "scenario": st.session_state[key] = "Mixed"

def get_row(mid, scenario):
    df = csv_data.get(mid)
    if df is None: return None
    fi = fault_idx.get(mid, {})
    i = st.session_state.row
    if scenario == "Normal": il = fi.get("normal", [])
    elif scenario == "Early Faults": il = fi.get("early", [])
    elif scenario == "Critical Faults": il = fi.get("critical", [])
    elif scenario == "Sensor Faults": il = fi.get("sensor", [])
    else:
        cp = i % 8
        if cp < 3: il = fi.get("normal", [])
        elif cp < 5: il = fi.get("early", [])
        elif cp < 7: il = fi.get("critical", [])
        else: il = fi.get("sensor", [])
    if il:
        si = (i // 8 if scenario == "Mixed" else i) % len(il)
        return df.iloc[il[si]]
    return df.iloc[i % len(df)]

def predict_machine(mid):
    info = MACHINES[mid]
    row = get_row(mid, st.session_state.scenario)
    if row is None: return None
    c = float(row.get("current_A", info["i"]*0.8))
    t = float(row.get("bearing_temp_C", 45))
    v = float(row.get("vibration_rms_mm_s", 2))
    k = float(row.get("vibration_kurtosis", 3))
    load = float(row.get("load_pct", 50))
    amb = float(row.get("ambient_temp", 25))
    true_state = str(row.get("machine_state", "normal"))
    true_fault = str(row.get("fault_type", "none"))
    true_sensor = str(row.get("sensor_health", "all_ok"))
    payload = {
        "machine_id": mid,
        "current_A": round(c, 2),
        "bearing_temp_C": round(t, 2),
        "vibration_rms_mm_s": round(v, 3),
        "vibration_kurtosis": round(k, 2),
        "load_pct": round(load, 1),
        "ambient_temp": round(amb, 1),
        "rolling_mean_current_30s": round(float(row.get("rolling_mean_current_30s", c)), 2),
        "rolling_std_vibration_30s": round(float(row.get("rolling_std_vibration_30s", 0)), 4),
        "temp_rate_of_change": round(float(row.get("temp_rate_of_change", 0)), 4),
        "current_rate_of_change": round(float(row.get("current_rate_of_change", 0)), 4)
    }
    base = {**payload, "true_state": true_state, "true_fault": true_fault,
            "true_sensor": true_sensor, "time": datetime.now().strftime("%H:%M:%S")}
    try:
        resp = requests.post(FLASK_URL+"/predict", json=payload, timeout=2)
        if resp.status_code == 200:
            return {**base, **resp.json(), "ml_available": True}
        base["ml_error"] = "HTTP %d" % resp.status_code
    except requests.RequestException as exc:
        base["ml_error"] = str(exc)
    # API failed: do NOT fall back to ground truth as a fake "prediction" --
    # that would make an API outage invisible (accuracy would read 100% and
    # every card would show MATCH by construction). ml_available=False is
    # the single flag every consumer below checks; ML fields stay unset.
    return {**base, "ml_available": False, "machine_health": None,
            "fault_type": None, "sensor_health": None,
            "health_confidence": None, "fault_confidence": None,
            "sensor_confidence": None}

def gen_waveform(mid, ft):
    """Illustrative waveform for a fault type's typical shape -- regenerated
    fresh each rerun, not derived from this machine's actual measured
    vibration signal. UI makes this explicit (see caption in the WAVEFORM tab)."""
    t = np.linspace(0, 0.128, 64)
    sf = MACHINES[mid]["rpm"] / 60.0
    if ft == "bearing_degradation":
        bpfo = sf * 3.607  # real 6208 datasheet value, see docs/PHYSICS.md
        s = np.random.normal(0,0.2,64)
        for x in np.arange(0, t[-1], 1/bpfo):
            idx = np.argmin(np.abs(t-x))
            s[idx] += np.random.uniform(2,4)
        return t, s, "BEARING DEFECT - BPFO=%.1fHz" % bpfo
    elif ft == "rotor_imbalance":
        s = 2.5*np.sin(2*np.pi*sf*t) + np.random.normal(0,0.2,64)
        return t, s, "IMBALANCE - 1X at %.1fHz" % sf
    elif ft == "cavitation":
        s = np.random.normal(0,0.3,64)
        for _ in range(5):
            p = np.random.randint(0,61)
            s[p:p+3] += np.random.uniform(2,5,3)
        return t, s, "CAVITATION - Random impacts"
    elif ft in ["overheating","overload"]:
        s = 1.5*np.sin(2*np.pi*sf*t) + np.random.normal(0,0.4,64)
        return t, s, "OVERLOAD - Elevated vibration"
    else:
        return t, np.random.normal(0,0.3,64), "NORMAL - Background noise"

# Increments on every rerun, and Streamlit reruns on any widget interaction,
# not just the refresh timer -- so trend-chart spacing is even in rows shown
# but not in real elapsed time if the user is clicking around. Would need a
# wall-clock gate to fix; not done here.
st.session_state.row += 1
cur = {}
for mid in MACHINES:
    d = predict_machine(mid)
    if d:
        cur[mid] = d
        st.session_state.readings[mid].append(d)
        if len(st.session_state.readings[mid]) > 300:
            st.session_state.readings[mid] = st.session_state.readings[mid][-300:]
        if d.get("machine_health") in ["early_fault","critical_fault"]:
            st.session_state.alerts.append({"TIME": d["time"], "TAG": MACHINES[mid]["name"],
                "TYPE": "MACHINE", "DETAIL": d["machine_health"]+" - "+d.get("fault_type","?")})
        # Check ml_available explicitly rather than just "!= all_ok":
        # None != "all_ok" is True in Python, so without this an API outage
        # would log a fake "SENSOR: None" alarm row every refresh.
        if d.get("ml_available") and d.get("sensor_health") != "all_ok":
            st.session_state.alerts.append({"TIME": d["time"], "TAG": MACHINES[mid]["name"],
                "TYPE": "SENSOR", "DETAIL": d["sensor_health"]})
        st.session_state.alerts = st.session_state.alerts[-50:]

st.sidebar.title("CONTROL PANEL")
if st.sidebar.button("PLANT OVERVIEW", use_container_width=True):
    st.session_state.page = "overview"
st.sidebar.markdown("---")
for mid, info in MACHINES.items():
    if st.sidebar.button(info["name"]+" - "+info["full"], key="n_"+mid, use_container_width=True):
        st.session_state.page = "detail"
        st.session_state.detail = mid
st.sidebar.markdown("---")
sc = st.sidebar.selectbox("SCENARIO", ["Mixed","Normal","Early Faults","Critical Faults","Sensor Faults"])
st.session_state.scenario = sc
rf = st.sidebar.selectbox("REFRESH", ["5s","10s","30s","Manual"], index=2)
rf_map = {"5s":5, "10s":10, "30s":30, "Manual":0}
rf_sec = rf_map[rf]
st.sidebar.markdown("---")
st.sidebar.caption("Scan #%d | %s" % (st.session_state.row, datetime.now().strftime("%H:%M:%S")))
try:
    r = requests.get(FLASK_URL+"/health", timeout=2)
    if r.status_code==200: st.sidebar.success("ML API: ONLINE")
    else: st.sidebar.error("ML API: ERROR")
except requests.RequestException: st.sidebar.error("ML API: OFFLINE")

# Only counts machines the ML API actually responded for -- a missing
# prediction is neither correct nor incorrect, so it's excluded rather
# than folded into either bucket.
scored = {mid: d for mid, d in cur.items() if d.get("ml_available")}
correct = sum(1 for d in scored.values() if d.get("machine_health")==d.get("true_state"))
total = len(scored)
st.sidebar.markdown("---")
st.sidebar.metric("ML ACCURACY", "%.0f%%" % (correct/total*100 if total>0 else 0))
st.sidebar.caption("Live sample of %d machine(s) per refresh -- one flipped "
                    "prediction swings this by ~33 points. A rolling demo "
                    "gauge, not a validated model metric." % max(total, 1))

st.markdown("<div class=\"main-header\">INDUSTRIAL SCADA - PLANT MONITORING</div>", unsafe_allow_html=True)

alarms = []
for mid, d in cur.items():
    if not d.get("ml_available"):
        alarms.append(("warn-bar", MACHINES[mid]["name"], "ML UNAVAILABLE - raw telemetry only"))
        continue
    h = d.get("machine_health","normal")
    s = d.get("sensor_health","all_ok")
    if h == "critical_fault": alarms.append(("alarm-bar", MACHINES[mid]["name"], "CRITICAL - "+d.get("fault_type","?")))
    elif h == "early_fault": alarms.append(("warn-bar", MACHINES[mid]["name"], "WARNING - "+d.get("fault_type","?")))
    if s != "all_ok": alarms.append(("warn-bar", MACHINES[mid]["name"], "SENSOR - "+s))  # warn, not ok-bar -- shouldn't look "nominal"
if alarms:
    for cls, tag, det in alarms:
        st.markdown("<div class=\"%s\">[%s] %s</div>" % (cls, tag, det), unsafe_allow_html=True)
else:
    st.markdown("<div class=\"ok-bar\">ALL SYSTEMS NOMINAL</div>", unsafe_allow_html=True)

if st.session_state.page == "overview":
    cols = st.columns(3)
    for idx, (mid, info) in enumerate(MACHINES.items()):
        d = cur.get(mid)
        if not d: continue
        with cols[idx]:
            st.markdown("#### %s" % info["name"])
            st.caption("%s | %s | %skW" % (info["full"], info["model"], info["kw"]))
            if not d.get("ml_available"):
                st.warning("ML UNAVAILABLE (%s)" % d.get("ml_error", "unreachable"))
            else:
                h = d.get("machine_health","normal")
                if h == "critical_fault": st.error("CRITICAL FAULT")
                elif h == "early_fault": st.warning("EARLY FAULT")
                else: st.success("RUNNING")
                ft = d.get("fault_type","none")
                if ft != "none": st.warning("Fault: %s" % ft)
                sh = d.get("sensor_health","all_ok")
                if sh != "all_ok": st.error("Sensor: %s" % sh)
            p = st.session_state.prev.get(mid, {})
            c1,c2,c3 = st.columns(3)
            with c1:
                dv = round(d["current_A"]-p.get("current_A",d["current_A"]),2)
                st.metric("I(A)", "%.1f" % d["current_A"], "%+.2f" % dv)
            with c2:
                dv = round(d["bearing_temp_C"]-p.get("bearing_temp_C",d["bearing_temp_C"]),2)
                st.metric("T(C)", "%.1f" % d["bearing_temp_C"], "%+.2f" % dv)
            with c3:
                dv = round(d["vibration_rms_mm_s"]-p.get("vibration_rms_mm_s",d["vibration_rms_mm_s"]),3)
                st.metric("V(mm/s)", "%.2f" % d["vibration_rms_mm_s"], "%+.3f" % dv)
            st.session_state.prev[mid] = {"current_A":d["current_A"],"bearing_temp_C":d["bearing_temp_C"],"vibration_rms_mm_s":d["vibration_rms_mm_s"]}
            if d.get("ml_available"):
                match = "MATCH" if d.get("machine_health")==d.get("true_state") else "MISMATCH"
                st.caption("ML: %s | True: %s | %s" % (h, d.get("true_state","?"), match))
            else:
                st.caption("ML: UNAVAILABLE | True: %s (sensor readings above are still real telemetry)" % d.get("true_state","?"))
    st.markdown("---")
    st.markdown("### TRENDS")
    tc = st.columns(3)
    for idx, (mid, info) in enumerate(MACHINES.items()):
        with tc[idx]:
            rd = st.session_state.readings[mid]
            if len(rd) > 2:
                vv = [r["vibration_rms_mm_s"] for r in rd[-60:]]
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=vv, mode="lines", line=dict(color="#00e676", width=2)))
                fig.update_layout(height=120, margin=dict(l=0,r=0,t=25,b=0), title=dict(text=info["name"], font=dict(size=11, color="#b0bec5")), xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.markdown("### ALARM LOG")
    al = st.session_state.alerts[-15:]
    if al: st.dataframe(pd.DataFrame(al), use_container_width=True, hide_index=True)
    else: st.markdown("<div class=\"ok-bar\">NO ALARMS</div>", unsafe_allow_html=True)

elif st.session_state.page == "detail":
    mid = st.session_state.detail
    info = MACHINES[mid]
    d = cur.get(mid)
    if not d: st.error("NO DATA"); st.stop()
    st.markdown("### %s - %s" % (info["name"], info["model"]))
    left, right = st.columns([2,3])
    with left:
        st.markdown("<div class=\"nameplate\">MODEL: %s<br>POWER: %skW<br>VOLTAGE: %s<br>I RATED: %sA<br>SPEED: %sRPM<br>EFF: %s pct</div>" % (info["model"],info["kw"],info["v"],info["i"],info["rpm"],info["eff"]), unsafe_allow_html=True)
        st.markdown("")
        ld = d.get("load_pct",50)
        sl = 2+(ld/100)*3
        rp = int(info["rpm"]*(1-sl/100))
        o1,o2,o3 = st.columns(3)
        with o1: st.metric("LOAD","%.0f%%" % ld)
        with o2: st.metric("RPM",rp)
        with o3: st.metric("SLIP","%.1f%%" % sl)
        st.markdown("##### ML CLASSIFICATION")
        if not d.get("ml_available"):
            st.markdown("<div class=\"warn-bar\">ML UNAVAILABLE (%s)</div>" %
                        d.get("ml_error", "API unreachable"), unsafe_allow_html=True)
            st.caption("Sensor readings elsewhere on this page are still real "
                       "telemetry -- only the model's classification is missing.")
        else:
            h = d.get("machine_health","normal")
            hc = d.get("health_confidence",0)
            if h=="critical_fault": st.markdown("<div class=\"alarm-bar\">HEALTH: CRITICAL (%.0f%%)</div>" % (hc*100), unsafe_allow_html=True)
            elif h=="early_fault": st.markdown("<div class=\"warn-bar\">HEALTH: WARNING (%.0f%%)</div>" % (hc*100), unsafe_allow_html=True)
            else: st.markdown("<div class=\"ok-bar\">HEALTH: NORMAL (%.0f%%)</div>" % (hc*100), unsafe_allow_html=True)
            st.progress(hc)
            ft = d.get("fault_type","none")
            fc = d.get("fault_confidence",0)
            if ft!="none": st.markdown("<div class=\"warn-bar\">FAULT: %s (%.0f%%)</div>" % (ft.upper(),fc*100), unsafe_allow_html=True)
            else: st.markdown("<div class=\"ok-bar\">NO FAULT (%.0f%%)</div>" % (fc*100), unsafe_allow_html=True)
            st.progress(fc)
            sh = d.get("sensor_health","all_ok")
            scc = d.get("sensor_confidence",0)
            if sh!="all_ok":
                st.markdown("<div class=\"alarm-bar\">SENSOR: %s (%.0f%%)</div>" % (sh.upper(),scc*100), unsafe_allow_html=True)
                st.error("%s sensor faulty. Others normal = SENSOR fault not MACHINE fault." % sh.replace("_fault","").upper())
            else: st.markdown("<div class=\"ok-bar\">SENSORS: OK (%.0f%%)</div>" % (scc*100), unsafe_allow_html=True)
            st.progress(scc)
            match = "MATCH" if d.get("machine_health")==d.get("true_state") else "MISMATCH"
            st.caption("TRUE: %s | %s | %s | %s" % (d.get("true_state","?"),d.get("true_fault","?"),d.get("true_sensor","?"),match))
    with right:
        t1,t2,t3,t4 = st.tabs(["TRENDS","GAUGES","WAVEFORM","ALARMS"])
        with t1:
            rd = st.session_state.readings[mid]
            if len(rd)>2:
                rdf = pd.DataFrame(rd[-300:])
                fc2 = go.Figure()
                fc2.add_trace(go.Scatter(y=rdf["current_A"],mode="lines",line=dict(color="#00bcd4",width=2)))
                fc2.add_hline(y=info["i"],line_dash="dash",line_color="#00e676")
                fc2.add_hline(y=info["i"]*1.15,line_dash="dash",line_color="#ff1744")
                fc2.update_layout(height=180,margin=dict(l=0,r=0,t=25,b=0),title="CURRENT(A)",showlegend=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(13,27,42,0.5)",font=dict(color="#b0bec5"))
                st.plotly_chart(fc2,use_container_width=True)
                ft2 = go.Figure()
                ft2.add_trace(go.Scatter(y=rdf["bearing_temp_C"],mode="lines",line=dict(color="#ff9800",width=2)))
                ft2.add_hline(y=65,line_dash="dash",line_color="#ffc107")
                ft2.add_hline(y=80,line_dash="dash",line_color="#ff1744")
                ft2.update_layout(height=180,margin=dict(l=0,r=0,t=25,b=0),title="TEMP(C)",showlegend=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(13,27,42,0.5)",font=dict(color="#b0bec5"))
                st.plotly_chart(ft2,use_container_width=True)
                z = info["iso"]
                fv = go.Figure()
                fv.add_hrect(y0=0,y1=z["A"],fillcolor="rgba(0,230,118,0.08)",line_width=0)
                fv.add_hrect(y0=z["A"],y1=z["B"],fillcolor="rgba(255,193,7,0.08)",line_width=0)
                fv.add_hrect(y0=z["B"],y1=z["C"],fillcolor="rgba(255,152,0,0.08)",line_width=0)
                fv.add_hrect(y0=z["C"],y1=15,fillcolor="rgba(255,23,68,0.08)",line_width=0)
                fv.add_trace(go.Scatter(y=rdf["vibration_rms_mm_s"],mode="lines",line=dict(color="#e040fb",width=2)))
                fv.update_layout(height=180,margin=dict(l=0,r=0,t=25,b=0),title="VIB(mm/s)-ISO10816",showlegend=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(13,27,42,0.5)",font=dict(color="#b0bec5"))
                st.plotly_chart(fv,use_container_width=True)
        with t2:
            g1,g2,g3 = st.columns(3)
            with g1:
                val = d.get("current_A",0) or 0
                axmax, axmin = max(info["i"]*2, abs(val)*1.1), min(0, val*1.1)
                fig=go.Figure(go.Indicator(mode="gauge+number",value=val,title={"text":"I(A)","font":{"color":"#b0bec5"}},number={"font":{"color":"#e0e0e0"}},gauge={"axis":{"range":[axmin,axmax]},"bar":{"color":"#00bcd4"},"bgcolor":"#0d1b2a","steps":[{"range":[0,info["i"]],"color":"rgba(0,230,118,0.2)"},{"range":[info["i"],info["i"]*1.15],"color":"rgba(255,193,7,0.2)"},{"range":[info["i"]*1.15,axmax],"color":"rgba(255,23,68,0.2)"}]}))
                fig.update_layout(height=220,margin=dict(l=15,r=15,t=40,b=0),paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig,use_container_width=True)
            with g2:
                val = d.get("bearing_temp_C",0) or 0
                axmax, axmin = max(120, val*1.1), min(0, val*1.1)
                fig=go.Figure(go.Indicator(mode="gauge+number",value=val,title={"text":"T(C)","font":{"color":"#b0bec5"}},number={"font":{"color":"#e0e0e0"}},gauge={"axis":{"range":[axmin,axmax]},"bar":{"color":"#ff9800"},"bgcolor":"#0d1b2a","steps":[{"range":[0,65],"color":"rgba(0,230,118,0.2)"},{"range":[65,80],"color":"rgba(255,193,7,0.2)"},{"range":[80,axmax],"color":"rgba(255,23,68,0.2)"}]}))
                fig.update_layout(height=220,margin=dict(l=15,r=15,t=40,b=0),paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig,use_container_width=True)
            with g3:
                val = d.get("vibration_rms_mm_s",0) or 0
                axmax, axmin = max(15, abs(val)*1.1), min(0, val*1.1)
                fig=go.Figure(go.Indicator(mode="gauge+number",value=val,title={"text":"V(mm/s)","font":{"color":"#b0bec5"}},number={"font":{"color":"#e0e0e0"}},gauge={"axis":{"range":[axmin,axmax]},"bar":{"color":"#e040fb"},"bgcolor":"#0d1b2a","steps":[{"range":[0,info["iso"]["A"]],"color":"rgba(0,230,118,0.2)"},{"range":[info["iso"]["A"],info["iso"]["B"]],"color":"rgba(255,193,7,0.2)"},{"range":[info["iso"]["B"],info["iso"]["C"]],"color":"rgba(255,152,0,0.2)"},{"range":[info["iso"]["C"],axmax],"color":"rgba(255,23,68,0.2)"}]}))
                fig.update_layout(height=220,margin=dict(l=15,r=15,t=40,b=0),paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig,use_container_width=True)
        with t3:
            ftype = d.get("fault_type","none")
            tw,sig,lab = gen_waveform(mid, ftype)
            clr = "#00e676" if ftype=="none" else "#ff1744"
            fw = go.Figure()
            fw.add_trace(go.Scatter(x=tw*1000,y=sig,mode="lines",line=dict(color=clr,width=1.5),fill="tozeroy"))
            fw.update_layout(height=280,margin=dict(l=0,r=0,t=30,b=0),title="WAVEFORM (500Hz) - ILLUSTRATIVE",xaxis_title="ms",yaxis_title="mm/s",showlegend=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(13,27,42,0.5)",font=dict(color="#b0bec5"))
            st.plotly_chart(fw,use_container_width=True)
            st.caption(lab)
            st.caption("Illustrative shape for this fault type, regenerated each refresh -- "
                       "not the actual measured signal (kurtosis below IS the real value).")
            st.caption("Kurtosis: %.2f" % d.get("vibration_kurtosis",3))
        with t4:
            ma = [a for a in st.session_state.alerts if a["TAG"]==info["name"]]
            if ma: st.dataframe(pd.DataFrame(ma[-20:]),use_container_width=True,hide_index=True)
            else: st.markdown("<div class=\"ok-bar\">NO ALARMS</div>",unsafe_allow_html=True)

if rf_sec > 0:
    time.sleep(rf_sec)
    st.rerun()
