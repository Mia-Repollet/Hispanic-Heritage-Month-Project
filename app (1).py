import io
import re
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

st.set_page_config(page_title="Hispanic Heritage Data Lab 2.0", page_icon="📈", layout="wide")
x = sp.Symbol("x", real=True)
TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

CONTEXTS = {
    "Business": ("Revenue", "time", "revenue", "year", "USD", "USD/year", "Revenue can rise or fall over time; the derivative is the rate of revenue change."),
    "Engineering": ("Position", "time", "position", "seconds", "meters", "meters/second", "The first derivative of position is velocity; the second derivative is acceleration."),
    "Medical": ("Drug concentration", "time", "concentration", "hours", "mg/L", "mg/L/hour", "Concentration changes over time; interpret any model as educational, not medical guidance."),
    "Sports science": ("Position", "time", "position", "seconds", "meters", "meters/second", "Position's first derivative is velocity and second derivative is acceleration."),
    "Biology": ("Population", "time", "population", "days", "organisms", "organisms/day", "The derivative describes population growth or decline."),
    "Bioengineering": ("Cell concentration", "time", "cell concentration", "hours", "cells/mL", "cells/mL/hour", "The derivative describes the changing cell concentration."),
    "Economic forecasting": ("Price index", "time", "index", "year", "index points", "points/year", "The derivative describes the rate of change in the index."),
    "Weather": ("Temperature", "time", "temperature", "hours", "°C", "°C/hour", "The derivative describes warming or cooling."),
    "Custom": ("Output", "input", "y", "x-units", "output units", "output units/input unit", "Interpret the rate of change using your chosen variables and units.")
}
SUGGESTIONS = {
    "Business": "a*x**2 + b*x + c", "Engineering": "a*x**2 + b*x + c",
    "Medical": "a*exp(-b*x) + c", "Sports science": "a*sin(b*x) + c",
    "Biology": "a*exp(b*x)", "Bioengineering": "a*exp(-b*x) + c",
    "Economic forecasting": "a*x + b", "Weather": "a*sin(b*x) + c", "Custom": "a*x**2 + b*x + c"
}
DEFAULTS = {"a": 1.0, "b": 0.2, "c": 0.0}
WB_INDICATORS = {
    "GDP per capita (current US$)": "NY.GDP.PCAP.CD",
    "Population, total": "SP.POP.TOTL",
    "Life expectancy at birth (years)": "SP.DYN.LE00.IN",
    "CO2 emissions (metric tons per capita)": "EN.ATM.CO2E.PC",
    "School enrollment, secondary (%)": "SE.SEC.ENRR"
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country_code, indicator_code):
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?format=json&per_page=200"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        raise ValueError("The data provider returned no observations for this selection.")
    rows = [{"year": int(row["date"]), "value": row["value"]} for row in payload[1] if row.get("value") is not None]
    df = pd.DataFrame(rows).drop_duplicates("year").sort_values("year")
    if df.empty:
        raise ValueError("No non-missing observations were returned.")
    return df

def parse_function(text):
    local = {"x": x, "pi": sp.pi, "E": sp.E, "e": sp.E, "sin": sp.sin, "cos": sp.cos,
             "tan": sp.tan, "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
             "abs": sp.Abs}
    try:
        expr = parse_expr(text, local_dict=local, transformations=TRANSFORMS, evaluate=True)
        if not expr.has(x) and not expr.is_number:
            raise ValueError("Include x as the independent variable.")
        if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            raise ValueError("The expression contains an undefined or infinite constant.")
        return expr
    except Exception as exc:
        raise ValueError("Could not parse the function. Use x, numbers, + - * /, powers (^ or **), and functions such as sin(x), exp(x), log(x), sqrt(x).") from exc

def safe_eval(expr, vals):
    fn = sp.lambdify(x, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        out = np.asarray(fn(vals), dtype=float)
    if out.shape == ():
        out = np.full(np.asarray(vals).shape, float(out))
    out[~np.isfinite(out)] = np.nan
    return out

def roots_numeric(expr, lo, hi, samples=1800):
    fn = sp.lambdify(x, expr, modules=["numpy"])
    grid = np.linspace(lo, hi, samples)
    vals = safe_eval(expr, grid)
    roots = []
    for i in range(len(grid)-1):
        a, b, ya, yb = grid[i], grid[i+1], vals[i], vals[i+1]
        if not (np.isfinite(ya) and np.isfinite(yb)): continue
        if abs(ya) < 1e-7: roots.append(a)
        elif ya * yb < 0:
            left, right = a, b
            for _ in range(45):
                mid = (left+right)/2
                ym = float(safe_eval(expr, np.array([mid]))[0])
                if not np.isfinite(ym): break
                if ya*ym <= 0: right = mid
                else: left, ya = mid, ym
            roots.append((left+right)/2)
    clean = []
    for v in roots:
        if not clean or abs(v-clean[-1]) > 1e-3: clean.append(float(v))
    return clean

def fmt(v, digits=5):
    if v is None: return "—"
    try:
        if not np.isfinite(float(v)): return "undefined"
        return f"{float(v):.{digits}g}"
    except Exception:
        return str(v)

def make_excel(summary, data, results):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(summary.items(), columns=["Item", "Value"]).to_excel(writer, index=False, sheet_name="Summary")
        data.to_excel(writer, index=False, sheet_name="Working data")
        results.to_excel(writer, index=False, sheet_name="Calculated years")
    return buf.getvalue()

def make_pdf(summary, data, results, chart_png=None):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [Paragraph("Hispanic Heritage Data Lab 2.0 — Model Report", styles["Title"]), Spacer(1, 10)]
    for k, v in summary.items():
        story.append(Paragraph(f"<b>{str(k)}</b>: {str(v)}", styles["BodyText"]))
    story.append(Spacer(1, 10))
    if chart_png:
        story.append(RLImage(io.BytesIO(chart_png), width=480, height=260))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Data preview", styles["Heading2"]))
    preview = data.head(12).copy()
    table_data = [list(preview.columns)] + preview.astype(str).values.tolist()
    t = Table(table_data, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
    story.append(t)
    story.append(Spacer(1, 10))
    story.append(Paragraph("Calculated-year results", styles["Heading2"]))
    if not results.empty:
        r = results.head(30)
        t2 = Table([list(r.columns)] + r.astype(str).values.tolist(), repeatRows=1)
        t2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(t2)
    doc.build(story)
    return buf.getvalue()

st.title("📈 Hispanic Heritage Data Lab 2.0")
st.caption("Explore real-world data, fit models, and investigate functions. Educational tool; models do not establish causes or guarantee forecasts.")
with st.sidebar:
    st.header("1 · Data and model")
    country = st.selectbox("Country", ["United States", "Mexico", "Spain", "Colombia", "Argentina", "Puerto Rico", "Canada", "Brazil", "India", "Japan"],
                           format_func=lambda z: z)
    country_codes = {"United States":"USA","Mexico":"MEX","Spain":"ESP","Colombia":"COL","Argentina":"ARG","Puerto Rico":"PRI","Canada":"CAN","Brazil":"BRA","India":"IND","Japan":"JPN"}
    indicator_label = st.selectbox("World Bank indicator", list(WB_INDICATORS))
    degree = st.slider("Polynomial degree", 1, 6, 2)
    sample_step = st.slider("Sample increment (years)", 1, 10, 2)
    refresh = st.button("Refresh data")
    if refresh:
        fetch_world_bank.clear()
    st.divider()
    st.header("2 · Course and context")
    course = st.selectbox("Course mode", ["Honors Algebra 2", "Precalculus", "Calculus"])
    context = st.selectbox("Context", list(CONTEXTS))
    use_fit = st.checkbox("Use fitted data function for analysis", value=True)
    suggested = SUGGESTIONS[context]
    custom_expr_text = st.text_input("Function in x (use ^ or ** for powers)", value=suggested)
    st.caption("Parameters: a, b, c. Edit them in the expression above, e.g. 2*x^2 - 3*x + 1.")
    context_title, x_label0, y_label0, x_unit0, y_unit0, deriv_unit0, context_help = CONTEXTS[context]
    with st.expander("Customize labels and interval", expanded=False):
        context_description = st.text_area("Context description", value=context_help)
        x_label = st.text_input("Independent variable", value=x_label0)
        y_label = st.text_input("Dependent variable", value=y_label0)
        x_unit = st.text_input("Independent-variable unit", value=x_unit0)
        y_unit = st.text_input("Dependent-variable unit", value=y_unit0)
        lo_user = st.number_input("Analysis interval start", value=0.0, step=1.0)
        hi_user = st.number_input("Analysis interval end", value=10.0, step=1.0)
    if 'context_description' not in locals():
        context_description, x_label, y_label, x_unit, y_unit, lo_user, hi_user = context_help, x_label0, y_label0, x_unit0, y_unit0, 0.0, 10.0

try:
    df = fetch_world_bank(country_codes[country], WB_INDICATORS[indicator_label]).sort_values("year")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    data_error = None
except Exception as exc:
    data_error = str(exc)
    df = pd.DataFrame({"year": [], "value": []})

if data_error:
    st.warning(f"Historical data could not be loaded: {data_error}. You can still explore a manually entered function below.")
else:
    st.success(f"Loaded {len(df)} observations for {country} — {indicator_label}.")

fit_ok = len(df) >= degree + 1 and df["year"].nunique() >= degree + 1
if fit_ok:
    base_year = int(df["year"].mean())
    t = df["year"].to_numpy(dtype=float) - base_year
    coeff = np.polyfit(t, df["value"].to_numpy(dtype=float), degree)
    fitted = np.polyval(coeff, t)
    resid = df["value"].to_numpy(dtype=float) - fitted
    ss_res = float(np.sum(resid**2)); ss_tot = float(np.sum((df["value"]-df["value"].mean())**2))
    r2 = 1 - ss_res/ss_tot if ss_tot else float("nan")
    mae = float(np.mean(np.abs(resid)))
    coef_str = " + ".join([f"({c:.10g})·(year − {base_year})^{degree-i}" for i,c in enumerate(coeff)])
    df = df.copy()
    df["fitted"] = fitted
    df["residual"] = resid
else:
    base_year = int(df["year"].mean()) if len(df) else None
    r2 = mae = float("nan")
    df = df.copy()
    if len(df): df["fitted"] = np.nan; df["residual"] = np.nan
    coef_str = "Unavailable: need at least degree + 1 distinct years."

tab_data, tab_fit, tab_calc, tab_func, tab_export = st.tabs(["Historical data", "Fit and compare", "Year calculator", "Function lab", "Export and deployment"])
with tab_data:
    st.subheader("Historical observations")
    if len(df):
        st.dataframe(df[["year","value"] + (["fitted","residual"] if fit_ok else [])], use_container_width=True, hide_index=True)
        st.caption(f"Coverage: {int(df.year.min())}–{int(df.year.max())}. Historical gaps are missing observations, not zeroes.")
        st.download_button("Download raw data CSV", df.to_csv(index=False).encode(), "historical_data.csv", "text/csv")
    else:
        st.info("No historical observations are currently available.")
with tab_fit:
    st.subheader("Centered polynomial fit")
    if fit_ok:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Degree", degree); c2.metric("R² · in-sample", fmt(r2)); c3.metric("MAE · in-sample", fmt(mae)); c4.metric("Observation count", len(df))
        st.markdown(f"**Base year:** {base_year}  \n**Equation:** fitted value = {coef_str}")
        st.write(f"**Historical range:** {int(df.year.min())}–{int(df.year.max())} · **Observed mean:** {fmt(df.value.mean())} {y_unit}")
        st.write("Residual = observed − fitted. In-sample metrics describe the data used to estimate the coefficients; they do not guarantee future accuracy.")
        if len(df) >= max(10, degree+5):
            hold_n = max(2, int(len(df)*0.2))
            train = df.iloc[:-hold_n]; test = df.iloc[-hold_n:]
            if train.year.nunique() >= degree+1:
                hbase = int(train.year.mean())
                hc = np.polyfit(train.year.to_numpy()-hbase, train.value.to_numpy(), degree)
                hp = np.polyval(hc, test.year.to_numpy()-hbase)
                st.write(f"**Chronological holdout MAE:** {fmt(np.mean(np.abs(test.value.to_numpy()-hp)))} (last {len(test)} observations held out; separate from in-sample metrics).")
        else:
            st.caption("Chronological holdout comparison omitted: too few observations for a useful split.")
        plot = go.Figure()
        plot.add_trace(go.Scatter(x=df.year, y=df.value, mode="markers", name="Observed data"))
        plot.add_trace(go.Scatter(x=df.year, y=df.fitted, mode="lines", name="Fitted at observed years"))
        years_smooth = np.linspace(df.year.min(), df.year.max(), max(200, len(df)*3))
        plot.add_trace(go.Scatter(x=years_smooth, y=np.polyval(coeff, years_smooth-base_year), mode="lines", name="Smooth fitted curve"))
        sample_years = np.arange(int(df.year.min()), int(df.year.max())+1, sample_step)
        plot.add_trace(go.Scatter(x=sample_years, y=np.polyval(coeff, sample_years-base_year), mode="markers", name=f"Model markers (step {sample_step})"))
        plot.update_layout(xaxis_title="Year", yaxis_title=f"{indicator_label} ({y_unit})", legend_title="Series", height=470)
        st.plotly_chart(plot, use_container_width=True)
    else:
        st.error(f"Cannot fit degree {degree}: need at least {degree+1} distinct years with non-missing values.")

with tab_calc:
    st.subheader("Calculate a fitted value by year")
    if not len(df) or not fit_ok:
        st.info("Load sufficient historical data and choose a supported degree to enable the fitted-year calculator.")
    else:
        year = st.number_input("Year (1950–2100)", min_value=1950, max_value=2100, value=max(1950,min(2100,int(df.year.max()))), step=1, key="single_year")
        if st.button("Calculate Value", type="primary"):
            estimate = float(np.polyval(coeff, year-base_year))
            inrange = int(df.year.min()) <= year <= int(df.year.max())
            st.write(f"**Year:** {year} · **Estimate:** {fmt(estimate)} {y_unit} · **Classification:** {'within fitted historical range' if inrange else 'outside fitted historical range (extrapolation)'}")
            if inrange:
                st.caption("This is the model's fitted value, not the actual observation. Compare it with the observed value in the historical table.")
            else:
                st.warning("Polynomial extrapolation can become implausible quickly. The estimate is not silently clipped and is not a validated prediction interval.")
        st.markdown("**Batch calculator**")
        batch_mode = st.radio("Input style", ["Year range", "Custom years"], horizontal=True)
        batch_years = []
        if batch_mode == "Year range":
            r1,r2,r3 = st.columns(3)
            start = r1.number_input("Start year", 1950, 2100, 2000, 1)
            end = r2.number_input("End year", 1950, 2100, 2030, 1)
            step = r3.number_input("Step (1–10)", 1, 10, 1, 1)
            if start <= end: batch_years = list(range(int(start), int(end)+1, int(step)))
            else: st.error("Start year must be less than or equal to end year.")
        else:
            raw = st.text_input("Comma-separated years", "2000, 2010, 2025")
            invalid = []
            for part in raw.split(","):
                part = part.strip()
                if not part: continue
                try:
                    v = int(part)
                    if 1950 <= v <= 2100: batch_years.append(v)
                    else: invalid.append(part)
                except ValueError: invalid.append(part)
            if invalid: st.error("Ignored invalid years (enter whole numbers from 1950 to 2100): " + ", ".join(invalid))
        if len(batch_years) > 500: st.error("Please request no more than 500 years at once."); batch_years = []
        batch_years = list(dict.fromkeys(batch_years))
        if batch_years:
            results = pd.DataFrame({"year": batch_years})
            results["estimate"] = np.polyval(coeff, results.year.to_numpy()-base_year)
            results["units"] = y_unit
            results["classification"] = results.year.apply(lambda z: "within fitted historical range" if df.year.min() <= z <= df.year.max() else "outside fitted historical range")
            actual_map = df.set_index("year")["value"]
            results["actual_observation"] = results.year.map(actual_map)
            st.dataframe(results, use_container_width=True, hide_index=True)
            st.caption("Batch estimates use the same centered-time transformation as the single-year calculator. Actual observations are shown separately.")
        else:
            results = pd.DataFrame(columns=["year","estimate","units","classification","actual_observation"])
            st.caption("Enter at least one valid requested year to calculate batch estimates.")
with tab_func:
    st.subheader("Function analysis lab")
    st.caption("The function lab is separate from the historical polynomial fit unless you select the fitted-data function. Mathematical results are exact where symbolic simplification succeeds; roots and plotted values are numerical approximations.")
    expr = None
    if use_fit and fit_ok:
        # Convert centered-time fit to a symbolic polynomial in x (year).
        expr = sum(sp.Float(float(c), 14) * (x-base_year)**(degree-i) for i,c in enumerate(coeff))
        st.info("Function source: polynomial fitted to the selected country and indicator. Here x represents calendar year.")
    else:
        try:
            expr = parse_function(custom_expr_text)
        except ValueError as exc:
            st.error(str(exc))
    if expr is not None:
        d1 = sp.diff(expr, x); d2 = sp.diff(d1, x)
        st.latex("f(x) = " + sp.latex(expr))
        if course == "Calculus":
            st.latex("f'(x) = " + sp.latex(d1))
            st.latex("f''(x) = " + sp.latex(d2))
        else:
            st.write("First derivative and second derivative are available in Calculus mode; this mode emphasizes graphs, domain/range, end behavior, and rates appropriate to the selected course.")
            if course == "Precalculus":
                st.latex("f'(x) = " + sp.latex(d1))
        if hi_user <= lo_user:
            st.error("Analysis interval end must be greater than its start.")
        else:
            lo, hi = float(lo_user), float(hi_user)
            xx = np.linspace(lo, hi, 900)
            yy = safe_eval(expr, xx)
            fig = go.Figure()
            if st.checkbox("Show function f(x)", True, key="show_f"):
                fig.add_trace(go.Scatter(x=xx, y=yy, name="f(x)", mode="lines", connectgaps=False))
            if course == "Calculus":
                if st.checkbox("Show first derivative f′(x)", False, key="show_d1"):
                    fig.add_trace(go.Scatter(x=xx, y=safe_eval(d1,xx), name="f′(x)", mode="lines", connectgaps=False))
                if st.checkbox("Show second derivative f″(x)", False, key="show_d2"):
                    fig.add_trace(go.Scatter(x=xx, y=safe_eval(d2,xx), name="f″(x)", mode="lines", connectgaps=False))
            fig.update_layout(xaxis_title=f"{x_label} ({x_unit})", yaxis_title=f"{y_label} ({y_unit})", height=450, legend_title="Curve")
            st.plotly_chart(fig, use_container_width=True)
            left,right = st.columns(2)
            with left:
                st.markdown("**Moving point and tangent**")
                point = st.slider("Current input x", float(lo), float(hi), float((lo+hi)/2), step=max((hi-lo)/1000, 0.001), key="point")
                fv = float(safe_eval(expr, np.array([point]))[0])
                dv = float(safe_eval(d1, np.array([point]))[0])
                ddv = float(safe_eval(d2, np.array([point]))[0])
                st.write(f"f({fmt(point)}) = {fmt(fv)} {y_unit}")
                st.write(f"f′({fmt(point)}) = {fmt(dv)} {y_unit}/{x_unit}")
                st.write(f"f″({fmt(point)}) = {fmt(ddv)} {y_unit}/{x_unit}²")
                if np.isfinite(fv) and np.isfinite(dv):
                    tangent = fv + dv*(xx-point)
                    figt = go.Figure()
                    figt.add_trace(go.Scatter(x=xx, y=yy, name="f(x)", mode="lines", connectgaps=False))
                    figt.add_trace(go.Scatter(x=xx, y=tangent, name="Tangent at current x", mode="lines"))
                    figt.add_trace(go.Scatter(x=[point], y=[fv], name="Moving point", mode="markers", marker_size=10))
                    figt.update_layout(xaxis_title=f"{x_label} ({x_unit})", yaxis_title=f"{y_label} ({y_unit})", height=360)
                    st.plotly_chart(figt, use_container_width=True)
                st.write(context_description)
                if "position" in y_label0.lower():
                    st.caption("For position models, f′ is velocity and f″ is acceleration.")
                else:
                    st.caption(f"f′ describes change in {y_label}; f″ describes how that rate changes. Units follow the displayed variables.")
            with right:
                st.markdown("**Average rate of change and secant**")
                a = st.slider("Interval start a", float(lo), float(hi), float(lo+(hi-lo)*0.25), step=max((hi-lo)/1000,0.001), key="sec_a")
                b = st.slider("Interval end b", float(lo), float(hi), float(lo+(hi-lo)*0.75), step=max((hi-lo)/1000,0.001), key="sec_b")
                fa, fb = float(safe_eval(expr,np.array([a]))[0]), float(safe_eval(expr,np.array([b]))[0])
                if b != a and np.isfinite(fa) and np.isfinite(fb):
                    avg = (fb-fa)/(b-a)
                    st.write(f"Average rate of change: ({fmt(fb)} − {fmt(fa)}) / ({fmt(b)} − {fmt(a)}) = **{fmt(avg)} {y_unit}/{x_unit}**")
                    sec = fa + avg*(xx-a)
                    figs = go.Figure()
                    figs.add_trace(go.Scatter(x=xx,y=yy,name="f(x)",mode="lines",connectgaps=False))
                    figs.add_trace(go.Scatter(x=xx,y=sec,name="Secant line",mode="lines"))
                    figs.add_trace(go.Scatter(x=[a,b],y=[fa,fb],name="Interval endpoints",mode="markers"))
                    figs.update_layout(xaxis_title=f"{x_label} ({x_unit})",yaxis_title=f"{y_label} ({y_unit})",height=360)
                    st.plotly_chart(figs,use_container_width=True)
                else:
                    st.warning("Choose distinct endpoints where the function is defined.")
            st.markdown("**Domain and interval analysis**")
            st.write("Domain restrictions: " + (", ".join(map(str, sp.calculus.util.continuous_domain(expr, x, sp.Interval(lo,hi)).args)) if False else "inspect denominators, logarithms, square roots, and plotted gaps; undefined values are not connected across gaps."))
            critical = roots_numeric(d1, lo, hi)
            inflect_candidates = roots_numeric(d2, lo, hi)
            st.write("Numerical critical-point candidates (f′ = 0): " + (", ".join(fmt(v) for v in critical) if critical else "none found in interval"))
            # Derivative sign test around each candidate
            extrema = []
            for v in critical:
                delta = min(1e-3, (hi-lo)/10000)
                before = float(safe_eval(d1,np.array([v-delta]))[0]) if v-delta >= lo else np.nan
                after = float(safe_eval(d1,np.array([v+delta]))[0]) if v+delta <= hi else np.nan
                kind = "local maximum" if before > 0 and after < 0 else "local minimum" if before < 0 and after > 0 else "not classified as an extremum by this sign test"
                extrema.append((v,kind,float(safe_eval(expr,np.array([v]))[0])))
            if extrema:
                st.write("Derivative sign-test classifications: " + "; ".join(f"x≈{fmt(v)}: {kind} (f≈{fmt(y)})" for v,kind,y in extrema))
            st.write("Inflection candidates (f″ = 0): " + (", ".join(fmt(v) for v in inflect_candidates) if inflect_candidates else "none found"))
            verified = []
            for v in inflect_candidates:
                delta=min(1e-3,(hi-lo)/10000)
                before=float(safe_eval(d2,np.array([v-delta]))[0]) if v-delta>=lo else np.nan
                after=float(safe_eval(d2,np.array([v+delta]))[0]) if v+delta<=hi else np.nan
                if np.isfinite(before) and np.isfinite(after) and before*after<0: verified.append(v)
            st.write("Verified inflection points (concavity changes numerically): " + (", ".join(f"({fmt(v)}, {fmt(float(safe_eval(expr,np.array([v]))[0]))})" for v in verified) if verified else "none confirmed in this interval"))
            grid_valid = np.isfinite(yy)
            if grid_valid.any():
                st.write(f"Approximate range on selected interval: [{fmt(np.nanmin(yy))}, {fmt(np.nanmax(yy))}] {y_unit}.")
                st.write(f"Endpoint values: f({fmt(lo)}) = {fmt(float(safe_eval(expr,np.array([lo]))[0]))}; f({fmt(hi)}) = {fmt(float(safe_eval(expr,np.array([hi]))[0]))}.")
            if course in ["Honors Algebra 2", "Precalculus"]:
                st.write("End behavior depends on the function family and its domain. For polynomials, the leading term determines end behavior; rational, logarithmic, exponential, and trigonometric functions can have asymptotes, restricted domains, or periodic behavior.")
            if course == "Precalculus":
                st.write("Degree and leading coefficient control polynomial end behavior; turning points are related to derivative zeros, and concavity is tracked by how average rates change.")
            if course == "Calculus":
                st.write("Critical-point classifications above use numerical derivative sign checks; endpoints must also be compared for absolute extrema on the closed interval. Numerical root searches can miss even-multiplicity roots or closely spaced features.")
            st.markdown("**Playback simulation**")
            st.caption("Use Play to step the input forward; playback is rerun-based and does not run continuously in the background.")
            playcol1, playcol2, playcol3 = st.columns(3)
            speed = playcol1.selectbox("Speed / step", ["Slow", "Medium", "Fast"], index=1)
            def advance_point():
                current = float(st.session_state.get("point", (lo+hi)/2))
                inc = {"Slow":(hi-lo)/100, "Medium":(hi-lo)/30, "Fast":(hi-lo)/10}[st.session_state.get("play_speed", "Medium")]
                st.session_state["point"] = lo if current + inc > hi else current + inc
            def reset_point():
                st.session_state["point"] = (lo+hi)/2
            st.session_state["play_speed"] = speed
            playcol2.button("▶ Play one step", on_click=advance_point)
            playcol3.button("↺ Reset point", on_click=reset_point)
            st.caption("Scaling note: graphs use the mathematical function's actual values and units; no physical scale is implied by the illustration.")
with tab_export:
    st.subheader("Export current selection")
    summary = {
        "Country": country, "Indicator": indicator_label, "Course mode": course, "Context": context,
        "Polynomial degree": degree, "Base year": base_year, "Equation": coef_str,
        "R-squared (in-sample)": fmt(r2), "MAE (in-sample)": fmt(mae),
        "Observation count": len(df), "Historical range": f"{int(df.year.min())}–{int(df.year.max())}" if len(df) else "unavailable",
        "Mean observed value": fmt(df.value.mean()) if len(df) else "unavailable",
        "Function lab expression": str(expr) if "expr" in locals() and expr is not None else custom_expr_text,
        "Data editing status": "Source data not edited in this session; no edit control is enabled."
    }
    results_export = pd.DataFrame(columns=["year","estimate","units","classification","actual_observation"])
    if "results" in locals() and isinstance(results, pd.DataFrame) and len(results):
        results_export = results.copy()
    st.download_button("Download current CSV", df.to_csv(index=False).encode("utf-8"), "hispanic_heritage_data.csv", "text/csv")
    st.download_button("Download Excel workbook (.xlsx)", make_excel(summary, df, results_export), "hispanic_heritage_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    # Build an export chart as a PNG when kaleido is available; PDF remains valid without it.
    chart_png = None
    try:
        if len(df) and fit_ok:
            pfig = go.Figure()
            pfig.add_trace(go.Scatter(x=df.year,y=df.value,mode="markers",name="Observed"))
            pfig.add_trace(go.Scatter(x=df.year,y=df.fitted,mode="lines",name="Fitted"))
            chart_png = pfig.to_image(format="png", width=900, height=480)
    except Exception:
        chart_png = None
    st.download_button("Download PDF report", make_pdf(summary, df, results_export, chart_png), "hispanic_heritage_report.pdf", "application/pdf")
    st.markdown("### Run locally")
    st.code("pip install -r requirements.txt\npython -m streamlit run app.py", language="bash")
    st.markdown("### Deploy")
    st.markdown("""
1. Create a GitHub repository containing `app.py` and `requirements.txt`; commit and push both files.
2. **Streamlit Community Cloud:** sign in, connect GitHub, choose the repository and branch, set the app entry point to `app.py`, then deploy.
3. **Render:** create a **Python Web Service** connected to the repository. Build command: `pip install -r requirements.txt`. Start command: `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`.
4. Check deployment logs if startup fails. Confirm the entry-point path, Python version, dependency installation, and that the port comes from `$PORT`.
""")
    st.markdown("### Verification checklist")
    st.markdown("""
- Compare source indicator, country, years, units, missing values, and observations against the World Bank API.
- Test a degree only when at least degree + 1 distinct years are available; check duplicate and missing years.
- Confirm the centered-time equation, in-sample R²/MAE, residuals, observation count, historical range, and mean.
- Compare single-year and batch estimates for identical years; test malformed years, out-of-range years, reversed ranges, and API/network failures.
- Test each context, course mode, function family, interval, graph toggle, point slider, secant endpoints, and reset/play controls.
- Open the CSV, Excel workbook, and PDF; verify they reflect the current selection and calculated-year results.
- For failures, inspect Streamlit and hosting logs; look for network timeouts, API rate limits, invalid expressions, missing dependencies, and missing/invalid environment ports.
""")
    st.markdown("### How the app works and its limitations")
    st.markdown("""
The app requests indicator observations from the World Bank API, removes missing values, fits a polynomial in centered time `t = year − base year`, and calculates residuals and in-sample fit metrics. The year calculator evaluates that same fitted transformation once, so years are not transformed twice. The function lab uses SymPy for parsing and symbolic derivatives, NumPy for numerical evaluation and root searches, and Plotly for interactive graphs. Exports are generated from the current selection.

A polynomial is a descriptive model, not evidence of causation. R² and MAE are in-sample unless explicitly labeled as chronological holdout. Polynomial extrapolation may be implausible; no validated 95% prediction interval is claimed. Numerical root finding and sign tests are approximations and may miss roots or discontinuities. This classroom version does not provide rigorous interval extrema proofs, fully general symbolic domain decomposition, or automatic editing/reset of source observations. Check the underlying data, units, assumptions, and limitations before defending an interpretation.
""")
