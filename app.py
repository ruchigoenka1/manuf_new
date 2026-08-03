import streamlit as st
import pandas as pd
import pulp
import plotly.express as px
import graphviz
import numpy as np
from datetime import datetime, timedelta
import streamlit_authenticator as stauth

# Import Metaheuristic libraries
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination

st.set_page_config(layout="wide", page_title="Advanced Job Scheduler")

# =========================================================================
# SECURE AUTHENTICATION MODULE (v0.3.0+)
# =========================================================================
config = {
    'credentials': {
        'usernames': {
            'admin': {
                'email': 'a@gamil.com',
                'name': 'System Admin',
                'password': '$2b$12$93MC4ONIi0.6QXjnL9uGveabXcSb1jCkauE4UiR68KeA5/0HRTyCK'
            },
            'founder_a': {
                'email': 'ashu@ashu.com',
                'name': 'Founder A',
                'password': '$2b$12$vbwv5qGinnaPy8qFWUQiseu6kK1y4bHu7W2CIuKQkLgNf7BilXCz2'
            },
            'consultant_b': {
                'email': 'consultant@example.com',
                'name': 'Consultant B',
                'password': '$2b$12$.CLG49Ex3t1kd0mjkTcp6eKL/gFSp/b1mTXrnH59hQa5KUKsrTbDS'
            },
            'analyst_c': {
                'email': 'analyst@example.com',
                'name': 'Analyst C',
                'password': 'PASTE_HASH_4_HERE'
            }
        }
    },
    'cookie': {
        'expiry_days': 1, # Set to 1 day so the user only logs in once a day
        'key': 'random_secret_signature_key_here', 
        'name': 'job_scheduler_cookie'
    }
}

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

try:
    authenticator.login()
except Exception as e:
    st.error(e)

if st.session_state.get("authentication_status") is False:
    st.error("Username/password is incorrect")
    st.stop()
elif st.session_state.get("authentication_status") is None:
    st.warning("Please enter your username and password to access the scheduler")
    st.stop()

# =========================================================================
# INITIALIZE SESSION STATE (Only runs if authenticated)
# =========================================================================
if "results_df" not in st.session_state:
    st.session_state.results_df = None
    st.session_state.makespan = None
    st.session_state.penalty_msg = ""

## --------------------------------------------------------
## 1. TITLE & MODE SELECTION
## --------------------------------------------------------
st.title("🗓️ Smart Job & Resource Scheduler")
st.markdown("Optimize production workflows and demand scheduling dynamically.")

with st.sidebar:
    st.write(f"Welcome, **{st.session_state.get('name', 'User')}**")
    authenticator.logout("Log Out", "sidebar")
    st.markdown("---")
    
    st.header("⚙️ Global Settings")
    default_date = datetime(2024, 1, 1)
    start_date = st.date_input("Project Start Date", default_date)

    scheduling_strategy = st.radio(
        "Scheduling Strategy Objective:",
        ("As Soon As Possible (ASAP)", "Just In Time / Close to Due Date")
    )

    solver_choice = st.radio(
        "Select Solving Engine:", 
        ("Optimizer", "Evolutionary Algorithm")
    )

    st.markdown("---")
    st.header("⏱️ Engine Parameters")

    if solver_choice == "Optimizer":
        time_limit = st.number_input(
            "Optimizer Time Limit (Seconds)", 
            min_value=10, max_value=1200, value=120, step=10,
            help="Limits how long the solver searches. Increase if you get timeout errors."
        )
    else:
        ga_generations = st.number_input(
            "No of Generation", 
            min_value=50, max_value=1000, value=100, step=50
        )
        ga_pop_size = st.number_input(
            "Size", 
            min_value=20, max_value=500, value=50, step=10
        )

## --------------------------------------------------------
## 2. STEP 1: DATA ENTRY (BASE RECIPES / ORDERS)
## --------------------------------------------------------
st.subheader("📋 Step 1: Define Job & Process Data")

default_data = pd.DataFrame([
    {"Job": "P1", "Process": "A", "Eligible_Resources": "R1", "Duration": 2, "Preceding_Process": ""},
    {"Job": "P1", "Process": "B", "Eligible_Resources": "R2", "Duration": 3, "Preceding_Process": "A"},
    {"Job": "P1", "Process": "C", "Eligible_Resources": "R2", "Duration": 2, "Preceding_Process": "B"},
    {"Job": "P1", "Process": "D", "Eligible_Resources": "R1", "Duration": 4, "Preceding_Process": "B"},
    {"Job": "P1", "Process": "E", "Eligible_Resources": "R1", "Duration": 2, "Preceding_Process": "C, D"},
    {"Job": "P2", "Process": "A", "Eligible_Resources": "R2, R1", "Duration": 3, "Preceding_Process": ""},
    {"Job": "P2", "Process": "B", "Eligible_Resources": "R1", "Duration": 2, "Preceding_Process": "A"},
])

uploaded_file = st.file_uploader("Upload your scheduling data (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            initial_data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            initial_data = pd.read_excel(uploaded_file)
        initial_data = initial_data.fillna("")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        initial_data = default_data
else:
    initial_data = default_data

df_input = st.data_editor(initial_data, num_rows="dynamic", use_container_width=True)
valid_df = df_input[(df_input['Job'] != '') & (df_input['Process'] != '')].copy()

st.markdown("---")

## --------------------------------------------------------
## 3. STEP 2: TEMPORAL BOUNDS (DEADLINES)
## --------------------------------------------------------
unique_jobs_list = sorted(list(valid_df['Job'].unique()))

st.subheader("⏳ Step 2: Project-Specific Time Windows")
st.markdown("Set the earliest start day (delay) and the deadline (in days from the project start date) for each job.")
default_deadlines = pd.DataFrame({
    "Job": unique_jobs_list, 
    "Earliest Start Day": [0] * len(unique_jobs_list),
    "Deadline (Days)": [30] * len(unique_jobs_list)
})
df_deadlines = st.data_editor(default_deadlines, hide_index=True, use_container_width=True)

start_day_dict = dict(zip(df_deadlines['Job'], df_deadlines['Earliest Start Day']))
deadline_dict = dict(zip(df_deadlines['Job'], df_deadlines['Deadline (Days)']))

st.markdown("---")

## --------------------------------------------------------
## 4. STEP 3: VISUAL MAP GENERATION
## --------------------------------------------------------
st.subheader("🗺️ Step 3: Verify Process Flow")

def generate_single_job_flowchart(df, job_name):
    dot = graphviz.Digraph(comment=f'Process Flow {job_name}')
    dot.attr(bgcolor='white', rankdir='LR')
    job_data = df[df['Job'] == job_name]
    dot.node(job_name, job_name, shape='box', style='filled', fillcolor='#1E3A8A', fontcolor='white', color='#1E3A8A')
    
    for idx, row in job_data.iterrows():
        process = str(row['Process']).strip()
        proc_id = f"{job_name}_{process}"
        with dot.subgraph() as s:
            s.attr(rank='same') 
            s.node(proc_id, process, shape='box', style='rounded,filled', fillcolor='#E0F2FE', fontcolor='black', color='#2563EB')
            resources = [r.strip() for r in str(row['Eligible_Resources']).split(',') if r.strip()]
            if resources:
                resources_str = ", ".join(resources) 
                res_id = f"{proc_id}_all_resources" 
                s.node(res_id, resources_str, shape='triangle', style='filled', fillcolor='#F0F9FF', fontcolor='black', color='#60A5FA')
                s.edge(proc_id, res_id, arrowhead='none', style='dotted', color='#60A5FA')
        
        preceding_str = str(row['Preceding_Process']).strip()
        if preceding_str:
            preceding_list = [p.strip() for p in preceding_str.split(',') if p.strip()]
            for pred in preceding_list:
                pred_id = f"{job_name}_{pred}"
                dot.edge(pred_id, proc_id, color='#3B82F6', penwidth='2')
        else:
            dot.edge(job_name, proc_id, color='#3B82F6', penwidth='2')
    return dot

with st.expander("👁️ View Process Flow Map", expanded=False):
    if not valid_df.empty:
        tabs = st.tabs(unique_jobs_list)
        for idx, job_name in enumerate(unique_jobs_list):
            with tabs[idx]:
                try:
                    st.graphviz_chart(generate_single_job_flowchart(valid_df, job_name), use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render map for {job_name}.")

st.markdown("---")

## --------------------------------------------------------
## 5. STEP 4: CHANGEOVER MATRIX
## --------------------------------------------------------
st.subheader("🔄 Step 4: Changeover Matrix (Job-Process Level)")
task_ids = [f"{row['Job']}_{row['Process']}" for idx, row in valid_df.iterrows()]
default_changeover = pd.DataFrame(0, index=task_ids, columns=task_ids)

with st.expander("📝 Edit Process-Level Changeover Matrix", expanded=False):
    df_changeover = st.data_editor(default_changeover, use_container_width=True)

st.markdown("---")

## --------------------------------------------------------
## 6. DISPLAY FUNCTIONS
## --------------------------------------------------------
def render_gantt_charts(df):
    st.subheader("📊 Interactive Sequence Gantt Charts")
    
    def get_ordinal(n):
        if 11 <= (n % 100) <= 13: return str(n) + 'th'
        return str(n) + {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    
    df["Project_Day"] = df["Start_Day"].apply(lambda x: f"{get_ordinal(x)} Day") + " to " + df["End_Day"].apply(lambda x: f"{get_ordinal(x)} Day")
    
    blue_colors = ['#1E3A8A', '#2563EB', '#3B82F6', '#60A5FA', '#93C5FD', '#BFDBFE']

    fig_job = px.timeline(
        df, x_start="Start", x_end="Finish", y="Job", color="Resource", text="Process", 
        title="Timeline Grouped by Production Batches", height=450,
        hover_data={"Project_Day": True, "Start": True, "Finish": True},
        color_discrete_sequence=blue_colors
    )
    fig_job.update_yaxes(autorange="reversed")
    fig_job.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    fig_job.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
    fig_job.update_traces(textposition='inside', insidetextanchor='middle')
    fig_job.update_traces(hovertemplate='<b>%{y}</b><br>Process: %{text}<br>Dates: %{base} to %{x}<br>Timeline: %{customdata[0]}<extra></extra>')
    st.plotly_chart(fig_job, use_container_width=True)
    st.markdown("---")
    
    fig_res = px.timeline(
        df, x_start="Start", x_end="Finish", y="Resource", color="Job", text="Process", 
        title="Timeline Grouped by Resource Allocation", height=450,
        hover_data={"Project_Day": True, "Start": True, "Finish": True},
        color_discrete_sequence=blue_colors
    )
    fig_res.update_yaxes(autorange="reversed")
    fig_res.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    fig_res.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
    fig_res.update_traces(textposition='inside', insidetextanchor='middle')
    fig_res.update_traces(hovertemplate='<b>%{y}</b><br>Process: %{text}<br>Dates: %{base} to %{x}<br>Timeline: %{customdata[0]}<extra></extra>')
    st.plotly_chart(fig_res, use_container_width=True)
    st.markdown("---")
    
    st.subheader("🔎 Individual Job Breakdown")
    selected_job = st.selectbox("Select a specific job to view its detailed flow:", sorted(df['Job'].unique()))
    
    if selected_job:
        job_df = df[df['Job'] == selected_job]
        fig_ind = px.timeline(
            job_df, x_start="Start", x_end="Finish", y="Process", color="Resource", text="Process",
            title=f"Detailed Flow: {selected_job}",
            height=max(300, 100 + (len(job_df['Process'].unique()) * 40)),
            hover_data={"Project_Day": True, "Start": True, "Finish": True},
            color_discrete_sequence=blue_colors
        )
        fig_ind.update_yaxes(autorange="reversed")
        fig_ind.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        fig_ind.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
        fig_ind.update_traces(
            textposition='inside', 
            insidetextanchor='middle',
            hovertemplate='<b>Process: %{y}</b><br>Resource: %{customdata[1]}<br>Dates: %{base} to %{x}<br>Timeline: %{customdata[0]}<extra></extra>'
        )
        st.plotly_chart(fig_ind, use_container_width=True)

def display_scheduling_results(results_df, total_makespan, penalty_msg=""):
    st.success(f"✨ Schedule Found! Total overall duration: **{total_makespan} days**.")
    if penalty_msg: st.warning(penalty_msg)
        
    st.subheader("📅 Step 5: Project Completion & Deadline Status")
    job_summary = results_df.groupby("Job").agg(Finished_Day=("End_Day", "max"), Finish_Date=("Finish", "max")).reset_index()
    job_summary["Deadline (Days)"] = job_summary["Job"].apply(lambda x: int(deadline_dict.get(x, 999)))
    job_summary = job_summary[["Job", "Finish_Date", "Finished_Day", "Deadline (Days)"]]
    job_summary.rename(columns={"Finished_Day": "Total Days Taken", "Finish_Date": "Completion Date"}, inplace=True)
    
    def highlight_late_jobs(row):
        if row["Total Days Taken"] > row["Deadline (Days)"]:
            return ['background-color: rgba(255, 75, 75, 0.3); color: black; font-weight: bold;'] * len(row)
        return [''] * len(row)
    st.dataframe(job_summary.style.apply(highlight_late_jobs, axis=1), use_container_width=True, hide_index=True)
    
    with st.expander("🔍 View Detailed Task Schedule Table"):
        st.dataframe(results_df[["Job", "Process", "Resource", "Start_Day", "End_Day"]], use_container_width=True, hide_index=True)
    
    render_gantt_charts(results_df)

## --------------------------------------------------------
## 7. EXECUTION BLOCK
## --------------------------------------------------------
base_tasks = []
for idx, row in valid_df.iterrows():
    base_tasks.append({
        'id': f"{row['Job']}_{row['Process']}", 'job': row['Job'], 'process': row['Process'],
        'resources': [r.strip() for r in str(row['Eligible_Resources']).split(',') if r.strip()],
        'duration': int(row['Duration']), 'preceding': [p.strip() for p in str(row['Preceding_Process']).split(',') if p.strip()]
    })

if st.button(f"🚀 Run {solver_choice}", type="primary"):
    if not base_tasks:
        st.error("Please ensure you have inputted process recipes in Step 1.")
        st.stop()

    if solver_choice == "Optimizer":
        prob = pulp.LpProblem("Demand_Scheduling", pulp.LpMinimize)
        start_vars = {t['id']: pulp.LpVariable(f"start_{t['id']}", lowBound=0, cat='Integer') for t in base_tasks}
        end_vars = {t['id']: pulp.LpVariable(f"end_{t['id']}", lowBound=0, cat='Integer') for t in base_tasks}
        assign_vars = {(t['id'], r): pulp.LpVariable(f"assign_{t['id']}_{r}", cat='Binary') for t in base_tasks for r in t['resources']}
        makespan = pulp.LpVariable("Makespan", lowBound=0, cat='Integer')
        
        if scheduling_strategy == "As Soon As Possible (ASAP)":
            prob += makespan * 1000 + pulp.lpSum([end_vars[t['id']] for t in base_tasks])
        else:
            prob += pulp.lpSum([ (int(deadline_dict.get(t['job'], 999)) - end_vars[t['id']]) for t in base_tasks ])
        
        for t in base_tasks:
            prob += end_vars[t['id']] == start_vars[t['id']] + t['duration']
            prob += makespan >= end_vars[t['id']]
            prob += pulp.lpSum([assign_vars[(t['id'], r)] for r in t['resources']]) == 1
            
            earliest_start = int(start_day_dict.get(t['job'], 0))
            prob += start_vars[t['id']] >= earliest_start
            
            for pred in t['preceding']:
                pred_id = f"{t['job']}_{pred}"
                if pred_id in start_vars: prob += start_vars[t['id']] >= end_vars[pred_id]

        M = max(1000, max(list(deadline_dict.values())) * 3) if deadline_dict else 3000
        for i in range(len(base_tasks)):
            for j in range(i + 1, len(base_tasks)):
                t1, t2 = base_tasks[i], base_tasks[j]
                common_res = set(t1['resources']).intersection(set(t2['resources']))
                common_res.discard("INV") 
                if common_res:
                    y = pulp.LpVariable(f"seq_{t1['id']}_{t2['id']}", cat='Binary')
                    for r in common_res:
                        c12 = int(df_changeover.loc[t1['id'], t2['id']]) if t1['id'] in df_changeover.index and t2['id'] in df_changeover.columns else 0
                        c21 = int(df_changeover.loc[t2['id'], t1['id']]) if t2['id'] in df_changeover.index and t1['id'] in df_changeover.columns else 0
                        prob += start_vars[t2['id']] >= end_vars[t1['id']] + c12 - M * (3 - assign_vars[(t1['id'], r)] - assign_vars[(t2['id'], r)] - y)
                        prob += start_vars[t1['id']] >= end_vars[t2['id']] + c21 - M * (2 - assign_vars[(t1['id'], r)] - assign_vars[(t2['id'], r)] + y)

        for t in base_tasks: prob += end_vars[t['id']] <= int(deadline_dict.get(t['job'], 999))

        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=False)
        with st.spinner("Calculating exact optimal schedule..."): status = prob.solve(solver)
        
        if pulp.LpStatus[status] in ["Optimal", "Not Solved"] and start_vars[base_tasks[0]['id']].varValue is not None:
            results = []
            for t in base_tasks:
                sel_res = [r for r in t['resources'] if assign_vars[(t['id'], r)].varValue > 0.5][0]
                s_val, e_val = int(start_vars[t['id']].varValue), int(end_vars[t['id']].varValue)
                results.append({
                    "Job": t['job'], "Process": t['process'], "Resource": sel_res, "Duration": t['duration'],
                    "Start_Day": s_val, "End_Day": e_val,
                    "Start": pd.to_datetime(start_date) + timedelta(days=s_val), "Finish": pd.to_datetime(start_date) + timedelta(days=e_val)
                })
            st.session_state.results_df = pd.DataFrame(results)
            st.session_state.makespan = int(makespan.varValue)
            st.session_state.penalty_msg = ""
        else:
            st.session_state.results_df = None
            st.error("❌ No feasible schedule found. Deadlines might be too tight.")
            
    else: # Evolutionary Algorithm
        def decode_demand(priorities, t_list, s_dict, d_dict, c_df, strategy):
            sorted_idx = np.argsort(priorities)[::-1] if strategy == "Just In Time / Close to Due Date" else np.argsort(priorities)
            res_avail, res_last, task_ends, sched = {}, {}, {}, []
            for t in t_list: 
                for r in t['resources']: res_avail[r] = 0
            penalty = 0
            
            for idx in sorted_idx:
                t = t_list[idx]
                p_ready = max([task_ends.get(f"{t['job']}_{p}", 0) for p in t['preceding']] + [0])
                job_earliest_start = int(s_dict.get(t['job'], 0)) 
                
                best_s, best_r = float('inf'), None
                for r in t['resources']:
                    if r == "INV":
                        ps = max(p_ready, job_earliest_start)
                    else:
                        c_time = int(c_df.loc[res_last[r], t['id']]) if r in res_last and res_last[r] in c_df.index else 0
                        ps = max(p_ready, res_avail[r] + c_time, job_earliest_start) 
                    if ps < best_s: best_s, best_r = ps, r
                    
                end = best_s + t['duration']
                
                if best_r != "INV":
                    res_avail[best_r], res_last[best_r] = end, t['id']
                
                task_ends[t['id']] = end
                sched.append({
                    "Job": t['job'], "Process": t['process'], "Resource": best_r, "Duration": t['duration'],
                    "Start_Day": best_s, "End_Day": end,
                    "Start": pd.to_datetime(start_date) + timedelta(days=best_s), "Finish": pd.to_datetime(start_date) + timedelta(days=end)
                })
                
                target_dl = int(d_dict.get(t['job'], 999))
                if end > target_dl: 
                    penalty += (end - target_dl) * 1000 
                elif strategy == "Just In Time / Close to Due Date":
                    penalty += (target_dl - end) * 5 
                    
            return sched, max(task_ends.values()) if task_ends else 0, penalty

        class DemandGA(ElementwiseProblem):
            def __init__(self, tl, sd, dd, cf, strat): 
                super().__init__(n_var=len(tl), n_obj=1, xl=0, xu=1)
                self.tl, self.sd, self.dd, self.cf, self.strat = tl, sd, dd, cf, strat
            def _evaluate(self, x, out, *args, **kwargs): 
                _, ms, pen = decode_demand(x, self.tl, self.sd, self.dd, self.cf, self.strat)
                out["F"] = ms + pen if self.strat == "As Soon As Possible (ASAP)" else pen

        with st.spinner("Evolving demand schedule..."):
            res = minimize(DemandGA(base_tasks, start_day_dict, deadline_dict, df_changeover, scheduling_strategy), GA(pop_size=ga_pop_size), get_termination("n_gen", ga_generations), seed=1)
            best_sched, best_ms, final_pen = decode_demand(res.X, base_tasks, start_day_dict, deadline_dict, df_changeover, scheduling_strategy)
            
            msg = "⚠️ Deadlines unachievable within set bounds." if final_pen > 10000 else ""
            
            st.session_state.results_df = pd.DataFrame(best_sched)
            st.session_state.makespan = best_ms
            st.session_state.penalty_msg = msg

# =========================================================================
# PERSISTENT DISPLAY BLOCK
# =========================================================================
if st.session_state.results_df is not None:
    display_scheduling_results(st.session_state.results_df, st.session_state.makespan, st.session_state.penalty_msg)
