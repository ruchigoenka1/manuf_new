import streamlit as st
import pandas as pd
import pulp
import plotly.express as px
import graphviz
import numpy as np
from datetime import datetime, timedelta
import streamlit_authenticator as stauth
import io

# Import Metaheuristic librarie
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
                'email': 'ashutosh.goenka123@gmail.com',
                'name': 'System Admin',
                'password': '$2b$12$93MC4ONIi0.6QXjnL9uGveabXcSb1jCkauE4UiR68KeA5/0HRTyCK'
            },
            
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
# st.title("🗓️ Smart Job & Resource Scheduler")
# st.markdown("Optimize production workflows and demand scheduling dynamically.")

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
## 1. TITLE & MODE SELECTION
## --------------------------------------------------------
st.title("Inventory & Manufacturing App")
# st.markdown("Optimize production workflows and demand scheduling dynamically.")

# ... [Your existing Sidebar Code] ...

if 'seed_counter' not in st.session_state:
    st.session_state.seed_counter = 42

# Add the tabs declaration here:
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Production Planning", "Demand Histogram Simulator", "Demand Analysis", "Continous Review Policy Simulator", "Periodic Review policy Simulator", "Inventory Audit", "Inventory KPI"])


with tab1:

    ## --------------------------------------------------------
    ## 2. STEP 1: DATA ENTRY (BASE RECIPES / ORDERS)
    ## --------------------------------------------------------
    st.title("Production Planning")
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
        # Changed bgcolor to 'transparent' so it inherits Streamlit's default background
        dot.attr(bgcolor='transparent', rankdir='LR')
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
        
        # Calculate Project Start and generate custom X-axis ticks (Dates + Day Number)
        project_start = df['Start'].iloc[0] - pd.Timedelta(days=int(df['Start_Day'].iloc[0]))
        max_date = df['Finish'].max()
        days_span = (max_date - project_start).days
        
        if days_span <= 15:
            tick_freq = '1D'
        elif days_span <= 45:
            tick_freq = '3D'
        elif days_span <= 90:
            tick_freq = '7D'
        else:
            tick_freq = '14D'
            
        tick_dates = pd.date_range(start=project_start, end=max_date, freq=tick_freq)
        tick_vals = tick_dates
        # ADDED +1 to the day calculation for the X-axis to make it 1-indexed
        tick_text = [f"{d.strftime('%b %d')}<br>(Day {(d - project_start).days + 1})" for d in tick_dates]
    
        # Format hover string to include Dates with Day numbers
        # ADDED +1 to Start_Day, and used End_Day directly to represent the inclusive 1-indexed end day
        df["Formatted_Dates"] = df.apply(
            lambda row: f"{row['Start'].strftime('%b %d, %Y')} (Day {row['Start_Day'] + 1}) to {(row['Finish'] - pd.Timedelta(days=1)).strftime('%b %d, %Y')} (Day {row['End_Day']})", 
            axis=1
        )
        
        # Custom color palette matching the provided screenshot
        base_colors = [
            '#8CD4FF', # Light Blue (R1)
            '#0673DF', # Deep Blue (R2)
            '#FFB4B4', # Light Pink/Salmon (R3)
            '#FF2B2B', # Red (R4)
            '#83E883', # Light Green (R5)
            '#FFA020', # Extra: Orange 
            '#B05BFF', # Extra: Purple
            '#FFCC00'  # Extra: Yellow
        ]
        
        # Automatically append Plotly's extended Alphabet palette for any resources beyond the first 8, 
        # ensuring no exact duplicates from our base colors.
        extended_colors = base_colors + [color for color in px.colors.qualitative.Alphabet if color not in base_colors]
    
        fig_job = px.timeline(
            df, x_start="Start", x_end="Finish", y="Job", color="Resource", text="Process", 
            title="Timeline Grouped by Production Batches", height=450,
            hover_data={"Formatted_Dates": True, "Duration": True, "Start": True, "Finish": True},
            color_discrete_sequence=extended_colors
        )
        fig_job.update_yaxes(autorange="reversed")
        fig_job.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", bargap=0.2) 
        fig_job.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='#E0E0E0',
            tickvals=tick_vals, ticktext=tick_text
        )
        fig_job.update_traces(textposition='inside', insidetextanchor='middle', width=0.8) 
        fig_job.update_traces(hovertemplate='<b>%{y}</b><br>Process: %{text}<br>Duration: %{customdata[1]} Days<br>Dates: %{customdata[0]}<extra></extra>')
        st.plotly_chart(fig_job, use_container_width=True)
        st.markdown("---")
        
        fig_res = px.timeline(
            df, x_start="Start", x_end="Finish", y="Resource", color="Job", text="Process", 
            title="Timeline Grouped by Resource Allocation", height=450,
            hover_data={"Formatted_Dates": True, "Duration": True, "Start": True, "Finish": True},
            color_discrete_sequence=extended_colors
        )
        fig_res.update_yaxes(autorange="reversed")
        fig_res.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", bargap=0.2) 
        fig_res.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='#E0E0E0',
            tickvals=tick_vals, ticktext=tick_text
        )
        fig_res.update_traces(textposition='inside', insidetextanchor='middle', width=0.8)
        fig_res.update_traces(hovertemplate='<b>%{y}</b><br>Process: %{text}<br>Duration: %{customdata[1]} Days<br>Dates: %{customdata[0]}<extra></extra>')
        st.plotly_chart(fig_res, use_container_width=True)
        st.markdown("---")
        
        st.subheader("🔎 Individual Job Breakdown")
        selected_job = st.selectbox("Select a specific job to view its detailed flow:", sorted(df['Job'].unique()))
        
        if selected_job:
            job_df = df[df['Job'] == selected_job]
            
            fig_ind = px.timeline(
                job_df, x_start="Start", x_end="Finish", y="Process", color="Resource", text="Process",
                title=f"Detailed Flow: {selected_job}",
                height=max(450, 150 + (len(job_df['Process'].unique()) * 60)),
                hover_data={"Formatted_Dates": True, "Duration": True, "Resource": True, "Start": True, "Finish": True},
                color_discrete_sequence=extended_colors
            )
            fig_ind.update_yaxes(autorange="reversed")
            fig_ind.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", bargap=0.2) 
            fig_ind.update_xaxes(
                showgrid=True, gridwidth=1, gridcolor='#E0E0E0',
                tickvals=tick_vals, ticktext=tick_text
            )
            fig_ind.update_traces(
                textposition='inside', 
                insidetextanchor='middle',
                width=0.8,
                hovertemplate='<b>Process: %{y}</b><br>Resource: %{customdata[2]}<br>Duration: %{customdata[1]} Days<br>Dates: %{customdata[0]}<extra></extra>'
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
with tab2:
    st.header("Demand & Lead Time Analyzer")
    
    # --- HELPER FUNCTION FOR REUSABLE ANALYSIS & PLOTTING ---
    def render_analysis_and_distribution(data_df, column_name, default_threshold=40.0, key_suffix=""):
        """Renders the Probability/Coverage Analysis and Histogram with unique keys to prevent collisions."""
        st.subheader(f"Probability & Coverage Analysis: {column_name}")
        
        analysis_col1, analysis_col2 = st.columns(2)
        with analysis_col1:
            st.markdown(f"#### Threshold Lookup (Points Below X)")
            threshold = st.number_input(f"Enter {column_name} Threshold:", value=float(default_threshold), step=1.0, key=f"thresh_{column_name}{key_suffix}")
            count_below = len(data_df[data_df[column_name] < threshold])
            percent_below = (count_below / len(data_df)) * 100 if len(data_df) > 0 else 0
            st.metric(f"Chances of {column_name} < {threshold}", f"{percent_below:.1f}%")
            st.caption(f"There are {count_below} periods where {column_name.lower()} was less than {threshold}.")
    
        with analysis_col2:
            st.markdown("#### Percentile Lookup (Coverage Level)")
            target_perc = st.number_input("Enter Service Level % (e.g. 95):", min_value=0.0, max_value=100.0, value=95.0, step=1.0, key=f"perc_{column_name}{key_suffix}")
            val_at_perc = np.percentile(data_df[column_name], target_perc)
            st.metric(f"{column_name} at {target_perc}% Service Level", f"{int(val_at_perc)}")
            st.caption(f"To cover {target_perc}% of all periods, you need to account for a {column_name.lower()} of {int(val_at_perc)}.")
    
        st.subheader(f"Visual Distribution: {column_name}")
        num_bins = st.slider("Select Number of Bins:", 5, 50, 15, key=f"bins_{column_name}{key_suffix}")
        counts, bin_edges = np.histogram(data_df[column_name], bins=num_bins)
        bin_size = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1
    
        fig = px.histogram(data_df, x=column_name, template="plotly_white", color_discrete_sequence=['#4F8BF9'])
        fig.update_traces(xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=bin_size))
        fig.add_vline(x=threshold, line_dash="dot", line_color="#EF553B", line_width=2.5, annotation_text=f"Threshold ({threshold})", annotation_position="top left")
        fig.add_vline(x=val_at_perc, line_dash="dot", line_color="#00CC96", line_width=2.5, annotation_text=f"{target_perc}% Coverage ({int(val_at_perc)})", annotation_position="top right")
        fig.update_layout(bargap=0.1, xaxis_title=f"{column_name} Quantity", yaxis_title="Count of Periods")
        st.plotly_chart(fig, use_container_width=True)
        
        table_col1, table_col2 = st.columns([1, 1])
        with table_col1:
            st.markdown("#### 📋 Statistical Summary")
            summary_stats = data_df[column_name].describe().to_frame().T
            st.dataframe(summary_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']], use_container_width=True)
    
        with table_col2:
            st.markdown("#### Bin Frequency Table")
            pct_total = counts / len(data_df) * 100
            bin_df = pd.DataFrame({
                "Bin Range": [f"{int(bin_edges[i])} - {int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)],
                "Frequency (Count)": counts,
                "% of Total": pct_total.round(1),
                "Cum. Count": counts.cumsum(),
                "Cum. %": pct_total.cumsum().round(1)
            })
            st.dataframe(bin_df, use_container_width=True, hide_index=True)


    # --- 1. Base Demand Configuration ---
    st.subheader("1. Base Demand Configuration")
    
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        dist_type = st.selectbox("Distribution Type", ("Normal", "Poisson", "Uniform"), key="dist_p1")
    with col_b:
        avg_demand = st.number_input("Average Demand", min_value=1.0, value=100.0, key="avg_p1")
    with col_c:
        num_periods = st.number_input("Number of Periods", min_value=10, value=10000, key="periods_p1")
    with col_d:
        if dist_type == "Normal":
            variation = st.number_input("Std Dev (Variation)", min_value=0.1, value=15.0, key="v_norm")
        elif dist_type == "Uniform":
            variation = st.number_input("Range (+/-)", min_value=1.0, value=30.0, key="v_uni")
        else:
            st.markdown("<p style='padding-top:25px; color:gray;'>Poisson variation fixed by Mean.</p>", unsafe_allow_html=True)
    
    np.random.seed(42)
    if dist_type == "Normal":
        generated = np.random.normal(avg_demand, variation, num_periods)
        df_base = pd.DataFrame({'Base Demand': np.floor(np.clip(generated, 0, None))})
    elif dist_type == "Poisson":
        generated = np.random.poisson(avg_demand, num_periods)
        df_base = pd.DataFrame({'Base Demand': np.floor(np.clip(generated, 0, None))})
    else: # Uniform Distribution
        # Calculate discrete bounds and ensure demand does not drop below 0
        min_val = max(0, int(avg_demand - variation))
        max_val = int(avg_demand + variation)
        
        # np.random.randint is exclusive at the upper bound, so we add 1 to include max_val
        generated = np.random.randint(min_val, max_val + 1, num_periods)
        df_base = pd.DataFrame({'Base Demand': generated})


    
    df_base = pd.DataFrame({'Base Demand': np.floor(np.clip(generated, 0, None))})

    # Calculate Base CoV
    mean_val = float(df_base['Base Demand'].mean())
    std_val = float(df_base['Base Demand'].std())
    base_cov = (std_val / mean_val) if mean_val > 0 else 0.0

    st.divider()

    render_analysis_and_distribution(df_base, 'Base Demand', default_threshold=avg_demand * 0.8, key_suffix="_t2")

    st.divider()
    st.subheader("📊 Base Demand Volatility Analysis (CoV)")
    cov_col1, cov_col2 = st.columns([1, 2])
    with cov_col1:
        st.markdown("#### Formula")
        st.latex(r"CoV = \frac{\sigma}{\mu}")
        st.caption(r"Where $\sigma$ = Standard Deviation and $\mu$ = Mean")
    with cov_col2:
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Mean ($\mu$)", f"{mean_val:.2f}")
        m_col2.metric("Std Dev ($\sigma$)", f"{std_val:.2f}")
        m_col3.metric("Calculated CoV", f"{base_cov:.3f}")

    # --- 2. Projected Demand Analysis (Constant CoV) ---
    st.divider()
    st.subheader("2. Projected Demand Analysis (Constant CoV)")
    st.write(f"Project future demand scenarios while maintaining the current CoV of **{base_cov:.3f}**.")
    
    proj_avg_demand = st.number_input("Enter Projected Average Demand:", min_value=1.0, value=float(avg_demand * 1.2), key="proj_avg")
    proj_std = proj_avg_demand * base_cov
    
    # Generate projected data using Normal dist to respect the strict CoV lock
    proj_generated = np.random.normal(proj_avg_demand, proj_std, num_periods)
    df_proj = pd.DataFrame({'Projected Demand': np.floor(np.clip(proj_generated, 0, None))})
    
    render_analysis_and_distribution(df_proj, 'Projected Demand', default_threshold=proj_avg_demand * 0.8, key_suffix="_t2_proj")

    # --- 3. Lead Time Demand Analysis (For Reorder Point) ---
    st.divider()
    st.subheader("3. Lead Time Demand (Rolling Analysis)")
    st.write("By calculating the rolling sum of daily demand over your lead time, we can visualize the actual **Lead Time Demand**. The percentiles of this distribution directly represent your required **Reorder Point (ROP)** to prevent stockouts.")
    
    # Updated to select Demand Type instead of Lead Time Type
    demand_source = st.radio("Select Demand Data for Analysis:", ("Historical (Base Demand)", "Forecasted (Projected Demand)"), horizontal=True)
    
    lt_days = st.number_input("Lead Time (Days)", min_value=1, value=14, step=1, key="lt_days")
        
    # Choose the correct dataframe and average based on the user's selection
    if demand_source == "Historical (Base Demand)":
        target_df = df_base
        target_col = 'Base Demand'
        active_avg = avg_demand
    else:
        target_df = df_proj
        target_col = 'Projected Demand'
        active_avg = proj_avg_demand
        
    # Calculate Rolling Lead Time Demand using the selected Demand data
    rolling_ltd = target_df[target_col].rolling(window=int(lt_days)).sum().dropna()
    df_ltd = pd.DataFrame({'Lead Time Demand': rolling_ltd})
    
    # Render the analysis using the helper function
    # Default threshold automatically adjusts based on which average demand is being used
    default_rop = float(active_avg * lt_days)
    render_analysis_and_distribution(df_ltd, 'Lead Time Demand', default_threshold=default_rop, key_suffix="_t2_ltd")

    

   # --- 4. Continuous Review Inventory Simulator ---
    st.divider()
    st.subheader("4. Inventory Performance Simulator")
    st.write("Run a day-by-day inventory simulation to evaluate how your parameters perform against actual demand patterns.")
    
    # Select the demand stream to simulate against
    sim_source = st.radio("Simulation Data Source:", ("Historical (Base Demand)", "Forecasted (Projected Demand)"), horizontal=True, key="sim_source")
    
    if sim_source == "Historical (Base Demand)":
        sim_demand = df_base['Base Demand'].values
        default_avg = avg_demand
    else:
        sim_demand = df_proj['Projected Demand'].values
        default_avg = proj_avg_demand
    
    inv_col1, inv_col2, inv_col3 = st.columns(3)
    with inv_col1:
        calc_rop = st.number_input("Reorder Point (Units)", min_value=0.0, value=float(default_avg * lt_days * 1.2), key="sim_rop")
    with inv_col2:
        calc_lt = st.number_input("Average Lead Time (Days)", min_value=1, value=int(lt_days), key="sim_lt")
    with inv_col3:
        calc_q = st.number_input("Order Quantity (Units)", min_value=1.0, value=float(default_avg * 10), help="Amount ordered when ROP is hit.", key="sim_q")

    # Layout toggle for pipeline visibility
    include_pipeline = st.checkbox("Include Pipeline Inventory (On-Order) in ROP Trigger", value=True)

    if st.button("▶️ Run Simulation", type="primary"):
        # Initialize simulation with a stable starting balance
        current_inv = 1.25 * calc_rop
        pipeline_inv = 0
        arrivals = {} 
        
        total_demand = 0
        total_fulfilled = 0
        stockout_days = 0
        inv_levels = []
        
        # Tracking arrays for stockout visual markers
        stockout_x = []
        stockout_y = []
        
        # Day-by-day simulation loop
        for day, d in enumerate(sim_demand):
            # 1. Receive any orders arriving today
            if day in arrivals:
                current_inv += arrivals[day]
                pipeline_inv -= arrivals[day]
                
            # 2. Fulfill daily demand
            fulfilled = min(current_inv, d)
            current_inv -= fulfilled
            
            # 3. Log metrics
            total_demand += d
            total_fulfilled += fulfilled
            
            # Log stockout event if we could not fulfill all demand
            if d > fulfilled:
                stockout_days += 1
                stockout_x.append(day)
                stockout_y.append(current_inv)  # This will be 0
                
            inv_levels.append(current_inv)
            
            # 4. End of day review (Check if we need to order)
            trigger_level = current_inv + (pipeline_inv if include_pipeline else 0)
            if trigger_level <= calc_rop:
                arrivals[day + int(calc_lt)] = arrivals.get(day + int(calc_lt), 0) + calc_q
                pipeline_inv += calc_q
                
        # Calculate final KPIs
        fill_rate_pct = (total_fulfilled / total_demand) * 100 if total_demand > 0 else 0
        total_days = len(sim_demand)
        min_inv = min(inv_levels)
        max_inv = max(inv_levels)
        avg_inv = sum(inv_levels) / len(inv_levels) if inv_levels else 0
        
        st.markdown("#### 🏆 Simulation KPIs")
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        
        kpi1.metric(
            "Fill Rate", 
            f"{fill_rate_pct:.1f}%", 
            f"{int(total_fulfilled):,} / {int(total_demand):,} Units", 
            delta_color="off"
        )
        kpi2.metric(
            "Stockout Days", 
            f"{stockout_days:,}", 
            f"Out of {total_days:,} Total Days", 
            delta_color="off"
        )
        kpi3.metric("Minimum Inventory", f"{int(min_inv):,} Units")
        kpi4.metric("Maximum Inventory", f"{int(max_inv):,} Units")
        kpi5.metric("Average Inventory", f"{int(avg_inv):,} Units")
        
        # Plot the inventory curve using graph_objects for more layered control
        import plotly.graph_objects as go
        
        fig_inv = go.Figure()
        
        # Base inventory line
        fig_inv.add_trace(go.Scatter(
            x=list(range(len(inv_levels))), 
            y=inv_levels, 
            mode='lines',
            line=dict(color='#0673DF'),
            name='Units on Hand'
        ))
        
        # Stockout markers
        if stockout_x:
            fig_inv.add_trace(go.Scatter(
                x=stockout_x,
                y=stockout_y,
                mode='markers',
                marker=dict(symbol='x', color='red', size=8, line=dict(width=1, color='darkred')),
                name='Stockout Event'
            ))
            
        fig_inv.add_hline(y=calc_rop, line_dash="dot", line_color="#EF553B", annotation_text="Reorder Point", annotation_position="top left")
        fig_inv.add_hline(y=0, line_color="black")
        
        fig_inv.update_layout(
            title="Inventory Level Over Time", 
            xaxis_title="Simulation Day", 
            yaxis_title="Units on Hand",
            template="plotly_white", 
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_inv, use_container_width=True)

import io
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from scipy.stats import norm

with tab3:
    st.header("Custom Data Analyzer")
    
    # --- 1. Data Upload & Configuration ---
    st.subheader("1. Upload Historical Data")
    
    up_col1, up_col2 = st.columns([2, 1])
    
    df_base_t3 = None
    
    with up_col1:
        uploaded_file = st.file_uploader("Upload your historical demand file (.xlsx or .csv):", type=["xlsx", "csv"], key="uploader_t3")
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_upload = pd.read_csv(uploaded_file)
                else:
                    df_upload = pd.read_excel(uploaded_file)
                
                if 'Demand' in df_upload.columns:
                    # Extract and rename to match the helper function logic
                    df_base_t3 = df_upload[['Demand']].dropna().copy()
                    df_base_t3['Demand'] = pd.to_numeric(df_base_t3['Demand'], errors='coerce')
                    df_base_t3 = df_base_t3.dropna()
                    df_base_t3.columns = ['Base Demand']
                    st.success("✅ File successfully uploaded and parsed!")
                else:
                    st.error("❌ Invalid Format: Your file must contain a column named exactly **'Demand'**.")
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                
    with up_col2:
        st.markdown("#### 📋 Download Template")
        st.caption("Please match your data format to this template. The sheet must include a column header named **Demand**.")
        
        template_df = pd.DataFrame({'Demand': [120, 95, 110, 135, 80, 105, 115]})
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            template_df.to_excel(writer, index=False, sheet_name='Template')
        
        st.download_button(
            label="📥 Download Excel Template",
            data=buffer.getvalue(),
            file_name="demand_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_template_t3"
        )

    # --- Proceed only if data is successfully uploaded ---
    if df_base_t3 is not None:
        
        with st.expander("🔢 View Raw Data", expanded=False):
            st.dataframe(df_base_t3, use_container_width=True, height=200)

        # Calculate Base Stats
        mean_val_t3 = float(df_base_t3['Base Demand'].mean())
        std_val_t3 = float(df_base_t3['Base Demand'].std())
        base_cov_t3 = (std_val_t3 / mean_val_t3) if mean_val_t3 > 0 else 0.0
        num_periods_t3 = len(df_base_t3)

        # Render Base Analysis
        render_analysis_and_distribution(df_base_t3, 'Base Demand', default_threshold=mean_val_t3 * 0.8, key_suffix="_t3")

        st.divider()
        st.subheader("📊 Base Demand Volatility Analysis (CoV)")
        cov_col1_t3, cov_col2_t3 = st.columns([1, 2])
        with cov_col1_t3:
            st.markdown("#### Formula")
            st.latex(r"CoV = \frac{\sigma}{\mu}")
            st.caption(r"Where $\sigma$ = Standard Deviation and $\mu$ = Mean")
        with cov_col2_t3:
            m_col1_t3, m_col2_t3, m_col3_t3 = st.columns(3)
            m_col1_t3.metric("Mean ($\mu$)", f"{mean_val_t3:.2f}")
            m_col2_t3.metric("Std Dev ($\sigma$)", f"{std_val_t3:.2f}")
            m_col3_t3.metric("Calculated CoV", f"{base_cov_t3:.3f}")

        # --- 2. Projected Demand Analysis (Constant CoV) ---
        st.divider()
        st.subheader("2. Projected Demand Analysis (Constant CoV)")
        st.write(f"Project future demand scenarios while maintaining your historical CoV of **{base_cov_t3:.3f}**.")
        
        proj_avg_demand_t3 = st.number_input("Enter Projected Average Demand:", min_value=1.0, value=float(mean_val_t3 * 1.2), key="proj_avg_t3")
        proj_std_t3 = proj_avg_demand_t3 * base_cov_t3
        
        # Generate projected data matching the uploaded length
        np.random.seed(42)
        proj_generated_t3 = np.random.normal(proj_avg_demand_t3, proj_std_t3, num_periods_t3)
        df_proj_t3 = pd.DataFrame({'Projected Demand': np.floor(np.clip(proj_generated_t3, 0, None))})
        
        render_analysis_and_distribution(df_proj_t3, 'Projected Demand', default_threshold=proj_avg_demand_t3 * 0.8, key_suffix="_t3_proj")

        # --- 3. Lead Time Demand Analysis ---
        st.divider()
        st.subheader("3. Lead Time Demand (Rolling Analysis)")
        st.write("By calculating the rolling sum of daily demand over your lead time, we can visualize the actual **Lead Time Demand**. The percentiles of this distribution directly represent your required **Reorder Point (ROP)** to prevent stockouts.")
        
        demand_source_t3 = st.radio("Select Demand Data for Analysis:", ("Historical (Base Demand)", "Forecasted (Projected Demand)"), horizontal=True, key="lt_source_t3")
        
        lt_days_t3 = st.number_input("Lead Time (Days)", min_value=1, value=14, step=1, key="lt_days_t3")
            
        if demand_source_t3 == "Historical (Base Demand)":
            target_df_t3 = df_base_t3
            target_col_t3 = 'Base Demand'
            active_avg_t3 = mean_val_t3
        else:
            target_df_t3 = df_proj_t3
            target_col_t3 = 'Projected Demand'
            active_avg_t3 = proj_avg_demand_t3
            
        rolling_ltd_t3 = target_df_t3[target_col_t3].rolling(window=int(lt_days_t3)).sum().dropna()
        df_ltd_t3 = pd.DataFrame({'Lead Time Demand': rolling_ltd_t3})
        
        default_rop_t3 = float(active_avg_t3 * lt_days_t3)
        render_analysis_and_distribution(df_ltd_t3, 'Lead Time Demand', default_threshold=default_rop_t3, key_suffix="_t3_ltd")

        # --- 4. Continuous Review Inventory Simulator ---
        st.divider()
        st.subheader("4. Inventory Performance Simulator")
        st.write("Run a day-by-day inventory simulation to evaluate how your parameters perform against actual demand patterns.")
        
        sim_source_t3 = st.radio("Simulation Data Source:", ("Historical (Base Demand)", "Forecasted (Projected Demand)"), horizontal=True, key="sim_source_t3")
        
        if sim_source_t3 == "Historical (Base Demand)":
            sim_demand_t3 = df_base_t3['Base Demand'].values
            default_avg_sim_t3 = mean_val_t3
        else:
            sim_demand_t3 = df_proj_t3['Projected Demand'].values
            default_avg_sim_t3 = proj_avg_demand_t3
        
        inv_col1_t3, inv_col2_t3, inv_col3_t3 = st.columns(3)
        with inv_col1_t3:
            calc_rop_t3 = st.number_input("Reorder Point (Units)", min_value=0.0, value=float(default_avg_sim_t3 * lt_days_t3 * 1.2), key="sim_rop_t3")
        with inv_col2_t3:
            calc_lt_t3 = st.number_input("Average Lead Time (Days)", min_value=1, value=int(lt_days_t3), key="sim_lt_t3")
        with inv_col3_t3:
            calc_q_t3 = st.number_input("Order Quantity (Units)", min_value=1.0, value=float(default_avg_sim_t3 * 10), help="Amount ordered when ROP is hit.", key="sim_q_t3")

        include_pipeline_t3 = st.checkbox("Include Pipeline Inventory (On-Order) in ROP Trigger", value=True, key="pipe_t3")

        if st.button("▶️ Run Simulation", type="primary", key="run_sim_t3"):
            current_inv_t3 = 1.25 * calc_rop_t3
            pipeline_inv_t3 = 0
            arrivals_t3 = {} 
            
            total_demand_t3 = 0
            total_fulfilled_t3 = 0
            stockout_days_t3 = 0
            inv_levels_t3 = []
            
            stockout_x_t3 = []
            stockout_y_t3 = []
            
            for day, d in enumerate(sim_demand_t3):
                if day in arrivals_t3:
                    current_inv_t3 += arrivals_t3[day]
                    pipeline_inv_t3 -= arrivals_t3[day]
                    
                fulfilled_t3 = min(current_inv_t3, d)
                current_inv_t3 -= fulfilled_t3
                
                total_demand_t3 += d
                total_fulfilled_t3 += fulfilled_t3
                
                if d > fulfilled_t3:
                    stockout_days_t3 += 1
                    stockout_x_t3.append(day)
                    stockout_y_t3.append(current_inv_t3) 
                    
                inv_levels_t3.append(current_inv_t3)
                
                trigger_level_t3 = current_inv_t3 + (pipeline_inv_t3 if include_pipeline_t3 else 0)
                if trigger_level_t3 <= calc_rop_t3:
                    arrivals_t3[day + int(calc_lt_t3)] = arrivals_t3.get(day + int(calc_lt_t3), 0) + calc_q_t3
                    pipeline_inv_t3 += calc_q_t3
                    
            fill_rate_pct_t3 = (total_fulfilled_t3 / total_demand_t3) * 100 if total_demand_t3 > 0 else 0
            total_days_t3 = len(sim_demand_t3)
            min_inv_t3 = min(inv_levels_t3)
            max_inv_t3 = max(inv_levels_t3)
            avg_inv_t3 = sum(inv_levels_t3) / len(inv_levels_t3) if inv_levels_t3 else 0
            
            st.markdown("#### 🏆 Simulation KPIs")
            kpi1_t3, kpi2_t3, kpi3_t3, kpi4_t3, kpi5_t3 = st.columns(5)
            
            kpi1_t3.metric(
                "Fill Rate", 
                f"{fill_rate_pct_t3:.1f}%", 
                f"{int(total_fulfilled_t3):,} / {int(total_demand_t3):,} Units", 
                delta_color="off"
            )
            kpi2_t3.metric(
                "Stockout Days", 
                f"{stockout_days_t3:,}", 
                f"Out of {total_days_t3:,} Total Days", 
                delta_color="off"
            )
            kpi3_t3.metric("Minimum Inventory", f"{int(min_inv_t3):,} Units")
            kpi4_t3.metric("Maximum Inventory", f"{int(max_inv_t3):,} Units")
            kpi5_t3.metric("Average Inventory", f"{int(avg_inv_t3):,} Units")
            
            import plotly.graph_objects as go
            
            fig_inv_t3 = go.Figure()
            
            fig_inv_t3.add_trace(go.Scatter(
                x=list(range(len(inv_levels_t3))), 
                y=inv_levels_t3, 
                mode='lines',
                line=dict(color='#0673DF'),
                name='Units on Hand'
            ))
            
            if stockout_x_t3:
                fig_inv_t3.add_trace(go.Scatter(
                    x=stockout_x_t3,
                    y=stockout_y_t3,
                    mode='markers',
                    marker=dict(symbol='x', color='red', size=8, line=dict(width=1, color='darkred')),
                    name='Stockout Event'
                ))
                
            fig_inv_t3.add_hline(y=calc_rop_t3, line_dash="dot", line_color="#EF553B", annotation_text="Reorder Point", annotation_position="top left")
            fig_inv_t3.add_hline(y=0, line_color="black")
            
            fig_inv_t3.update_layout(
                title="Inventory Level Over Time", 
                xaxis_title="Simulation Day", 
                yaxis_title="Units on Hand",
                template="plotly_white", 
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_inv_t3, use_container_width=True)


# ==========================================
# TAB 4: CONTINUOUS REVIEW 
# ==========================================
with tab4:
    st.markdown(
        """
        <style>
        .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
            padding-top: 2rem;
        }
        [data-testid="column"]:first-child {
            border-right: 1px solid rgba(255, 255, 255, 0.1);
            padding-right: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.header("Continuous Review")
    st.divider()

    # Main split layout
    input_col, output_col = st.columns([1, 3])

    # ================================================
    # LEFT PANEL: INPUTS
    # ================================================
    with input_col:
        st.subheader("⚙️ Parameters")
        
        st.markdown("**Basic Settings**")
        opening_balance = st.number_input("Opening Balance", value=500, key="sim_ob_t4")
        unit_value = st.number_input("Value Per Unit", value=100, key="sim_vu_t4")
        num_days = st.slider("Simulation Days", 100, 2000, 365, key="sim_nd_t4")
        
        st.markdown("**Demand Settings**")
        avg_demand = st.number_input("Average Demand", value=25, key="sim_ad_t4")
        cov = st.number_input("Demand CoV", value=0.8, key="sim_cov_t4")
        
        if "demand_sequence_tab4" not in st.session_state:
            st.session_state.demand_sequence_tab4 = None

        if st.button("🔄 Generate New Demand", key="reset_dem_t4", use_container_width=True):
            st.session_state.demand_sequence_tab4 = None
            
        st.markdown("**Policy Settings**")
        # Fixed duplicate key here:
        lead_time = st.number_input("Lead Time (Days)", value=3, key="sim_lt_t4")
        reorder_point = st.number_input("Reorder Point", value=200, key="sim_rp_t4")
        order_qty = st.number_input("Order Quantity", value=300, key="sim_oq_t4")
        
        st.markdown("**Cost Metrics**")
        holding_cost_percent = st.number_input("Holding Cost (%)", value=20.0, key="sim_hc_t4")
        ordering_cost = st.number_input("Ordering Cost / Order", value=500, key="sim_oc_t4")

    # ================================================
    # BACKGROUND CALCULATIONS
    # ================================================
    holding_cost_rate = holding_cost_percent / 100
    std_demand = avg_demand * cov

    if st.session_state.demand_sequence_tab4 is None:
        st.session_state.demand_sequence_tab4 = np.maximum(
            0,
            np.random.normal(avg_demand, std_demand, num_days)
        ).round()

    demand = st.session_state.demand_sequence_tab4
    dates = pd.date_range(start="2024-01-01", periods=num_days)

    inventory = opening_balance
    pipeline_orders = []
    data = []

    for day in range(num_days):
        shipment_received = 0
        for order in pipeline_orders.copy():
            if order[0] == day:
                shipment_received += order[1]
                pipeline_orders.remove(order)

        opening = inventory
        inventory += shipment_received
        demand_today = demand[day]
        inventory -= demand_today

        if inventory < 0:
            inventory = 0

        pipeline_qty = sum(qty for arrival, qty in pipeline_orders)
        inventory_position = opening - demand_today + shipment_received + pipeline_qty
        new_order = 0

        if inventory_position < reorder_point:
            new_order = order_qty
            pipeline_orders.append((day + lead_time, order_qty))

        closing = inventory
        closing_with_pipeline = closing + sum(qty for arrival, qty in pipeline_orders)

        data.append([
            dates[day], opening, demand_today, shipment_received, pipeline_qty,
            inventory_position, new_order, closing, closing_with_pipeline
        ])

    df = pd.DataFrame(data, columns=[
        "Date", "Opening Balance", "Demand", "Shipment Received", "Pipeline Order",
        "Inventory Position", "New Order", "Closing Balance", "Closing Balance Including Pipeline"
    ])

    # KPI logic execution
    stockout_days = (df["Closing Balance"] == 0).sum()
    average_inventory = df["Closing Balance Including Pipeline"].mean()
    average_age_inventory = average_inventory / df["Demand"].mean() if df["Demand"].mean() > 0 else 0

    df["Blocked Working Capital"] = df["Inventory Position"] * unit_value
    average_working_capital = df["Blocked Working Capital"].mean()

    min_inventory = df["Closing Balance"].min()
    max_inventory = df["Closing Balance"].max()
    min_wc = df["Blocked Working Capital"].min()
    max_wc = df["Blocked Working Capital"].max()

    df["Inventory Value"] = df["Closing Balance Including Pipeline"] * unit_value
    df["Holding Cost"] = df["Inventory Value"] * holding_cost_rate / 365
    total_holding_cost = df["Holding Cost"].sum()

    number_of_orders = (df["New Order"] > 0).sum()
    total_ordering_cost = number_of_orders * ordering_cost
    total_inventory_cost = total_holding_cost + total_ordering_cost

    annual_demand = avg_demand * 365
    holding_cost_per_unit = unit_value * holding_cost_rate
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit) if holding_cost_per_unit > 0 else 0

    def simulate_inventory_cost(order_quantity):
        sim_inv = opening_balance
        sim_pipeline = []
        holding_cost_total = 0
        orders_count = 0

        for day in range(num_days):
            shipment_rec = 0
            for order in sim_pipeline.copy():
                if order[0] == day:
                    shipment_rec += order[1]
                    sim_pipeline.remove(order)

            sim_inv += shipment_rec
            dem_today = demand[day]
            sim_inv -= dem_today

            if sim_inv < 0:
                sim_inv = 0

            pip_qty = sum(qty for arrival, qty in sim_pipeline)
            inv_pos = sim_inv + pip_qty

            if inv_pos < reorder_point:
                sim_pipeline.append((day + lead_time, order_quantity))
                orders_count += 1

            close_w_pip = sim_inv + sum(qty for arrival, qty in sim_pipeline)
            inv_val = close_w_pip * unit_value
            hold_cost_today = inv_val * holding_cost_rate / 365
            holding_cost_total += hold_cost_today

        order_cost_tot = orders_count * ordering_cost
        return holding_cost_total + order_cost_tot

    cost_current_policy = simulate_inventory_cost(order_qty)
    cost_eoq_policy = simulate_inventory_cost(int(eoq))


    # ================================================
    # RIGHT PANEL: OUTPUTS & DASHBOARD
    # ================================================
    with output_col:
        
        # Matrix Collapsible Section 1: Core KPIs
        with st.expander("📊 View Core Inventory & Financial Metrics", expanded=True):
            st.markdown("#### Primary KPIs")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Stockout Days", stockout_days)
            c2.metric("Avg Age of Inventory", round(average_age_inventory, 1))
            c3.metric("Average Inventory", round(average_inventory, 0))
            c4.metric("Avg Working Capital", f"${round(average_working_capital, 0):,}")

            st.markdown("#### Inventory & Capital Ranges")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Minimum Inventory", round(min_inventory, 0))
            r2.metric("Maximum Inventory", round(max_inventory, 0))
            r3.metric("Min Working Capital", f"${round(min_wc, 0):,}")
            r4.metric("Max Working Capital", f"${round(max_wc, 0):,}")

            st.markdown("#### Cost Metrics Breakdown")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Total Holding Cost", f"${round(total_holding_cost, 0):,}")
            cc2.metric("Total Ordering Cost", f"${round(total_ordering_cost, 0):,}")
            cc3.metric("Total Inventory Cost", f"${round(total_inventory_cost, 0):,}")

        # Matrix Collapsible Section 2: Optimization
        with st.expander("💡 View EOQ & Policy Optimization", expanded=False):
            st.markdown("#### Economic Order Quantity (EOQ)")
            e1, e2 = st.columns(2)
            e1.metric("Economic Order Quantity (EOQ)", round(eoq, 0))
            e2.metric("Selected Order Quantity", order_qty)
            
            st.markdown("#### Policy Financial Comparison")
            k1, k2, k3 = st.columns(3)
            k1.metric("Cost with Current Policy", f"${round(cost_current_policy, 0):,}")
            k2.metric("Cost with EOQ Policy", f"${round(cost_eoq_policy, 0):,}")
            k3.metric("Savings Using EOQ", f"${round(cost_current_policy - cost_eoq_policy, 0):,}")


        # Main Behaviour Chart
        st.markdown("#### Inventory Behaviour")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance"], name="Closing Inventory"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance Including Pipeline"], name="Inventory Position"))
        fig.add_hline(y=reorder_point, line_dash="dash", annotation_text="Reorder Point")

        stockouts = df[df["Closing Balance"] == 0]
        fig.add_trace(go.Scatter(x=stockouts["Date"], y=stockouts["Closing Balance"], mode="markers", name="Stockout", marker=dict(color="red", size=9)))

        reorders = df[df["New Order"] > 0]
        fig.add_trace(go.Scatter(x=reorders["Date"], y=reorders["Closing Balance"], mode="markers", name="Reorder Trigger", marker=dict(color="green", symbol="triangle-up", size=10)))

        fig.add_hrect(y0=0, y1=reorder_point*0.5, fillcolor="red", opacity=0.08)
        fig.add_hrect(y0=reorder_point*0.5, y1=reorder_point, fillcolor="yellow", opacity=0.08)
        fig.add_hrect(y0=reorder_point, y1=df["Closing Balance Including Pipeline"].max()*1.2, fillcolor="green", opacity=0.05)
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
        fig.update_yaxes(rangemode="tozero")
        
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

        # Secondary Charts (Grid Layout)
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### Blocked Working Capital")
            fig_wc = px.line(df, x="Date", y="Blocked Working Capital")
            fig_wc.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=280)
            st.plotly_chart(fig_wc, use_container_width=True)

            st.markdown("#### Demand Distribution")
            fig_hist = px.histogram(df, x="Demand", nbins=20)
            fig_hist.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=280)
            st.plotly_chart(fig_hist, use_container_width=True)

        with chart_col2:
            st.markdown("#### Pipeline Orders")
            fig_pipeline = px.line(df, x="Date", y="Pipeline Order")
            fig_pipeline.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=280)
            st.plotly_chart(fig_pipeline, use_container_width=True)

            st.markdown("#### Orders Placed")
            orders = df[df["New Order"] > 0]
            fig_orders = px.scatter(orders, x="Date", y="New Order")
            fig_orders.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=280)
            st.plotly_chart(fig_orders, use_container_width=True)

        st.divider()

        # Deep Dives (Expanders to conserve vertical space)
        with st.expander("📊 View Interactive Waterfall Analysis & Raw Data"):
            st.markdown("#### Inventory Flow Waterfall")
            selected_day = st.slider("Select Day for Waterfall Analysis", 0, len(df)-1, 0, key="waterfall_slider_t4")
            row = df.iloc[selected_day]

            fig_waterfall = go.Figure(go.Waterfall(
                measure=["absolute", "relative", "relative", "total"],
                x=["Opening Balance", "Demand", "Shipment Received", "Closing Balance"],
                y=[row["Opening Balance"], -row["Demand"], row["Shipment Received"], row["Closing Balance"]]
            ))
            fig_waterfall.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_waterfall, use_container_width=True)
            
            st.markdown("#### Simulation Output Table")
            st.dataframe(df, use_container_width=True)


# ==========================================
# TAB 5: PERIODIC REVIEW
# ==========================================
with tab5:
    st.header("Periodic Review")
    
    st.markdown("""
    In a periodic review system, inventory is checked at fixed intervals. The strategy must account for the mechanical reality of the **Protection Interval**—the time from when an order is placed until the *next* order can be placed and received.
    """)

    # --- Action: Regenerate Demand Button ---
    btn_col1, btn_col2 = st.columns([1, 5])
    with btn_col1:
        if st.button("🔄 Generate New Demand", key="regen_demand_pr"):
            st.session_state.seed_counter += 1

    # --- 1. Baseline System Parameters Input ---
    st.subheader("1. Supply Chain Parameters & Recommended Baseline")
    
    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    with p_col1:
        pr_avg_demand = st.number_input("Avg Daily Demand", value=100.0, step=10.0)
        pr_std_dev = st.number_input("Demand Std Dev", value=15.0, step=5.0)
    with p_col2:
        review_period = st.number_input("Recommended Review (Days)", value=14, min_value=1, step=1)
        lead_time = st.number_input("Lead Time (Days)", value=7, min_value=1, step=1)
    with p_col3:
        target_service_level = st.slider("Target Service Level (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.1)
        z_score = norm.ppf(target_service_level / 100.0)
    with p_col4:
        unit_cost = st.number_input("Unit Cost ($)", value=50.0, step=5.0)
        ordering_cost = st.number_input("Ordering Cost ($/order)", value=250.0, step=50.0, key="baseline_oc")
    with p_col5:
        holding_cost_pct = st.number_input("Annual Holding Cost (%)", value=20.0, step=1.0)
        holding_cost_annual = unit_cost * (holding_cost_pct / 100.0)
        holding_cost_daily = holding_cost_annual / 365.0

    # Calculate Recommended Baseline Target
    protection_interval = review_period + lead_time
    expected_demand_pi = pr_avg_demand * protection_interval
    std_dev_pi = pr_std_dev * np.sqrt(protection_interval)
    safety_stock = z_score * std_dev_pi
    recommended_target = expected_demand_pi + safety_stock

    st.info(f"**Calculated Baseline Target:** {int(recommended_target)} Units (Accommodating a {review_period}-day review cycle and {lead_time}-day lead time).")

    # --- 2. Multi-Scenario Customization Setup ---
    st.divider()
    
    def sync_ordering_costs():
        for i in range(5):  # Max slider value is 5
            st.session_state[f"oc_key_{i}"] = st.session_state.baseline_oc

    head_col1, head_col2 = st.columns([2, 1])
    with head_col1:
        st.subheader("2. Multi-Scenario Strategy Comparison")
    with head_col2:
        st.write("") # Spacing
        st.button("📋 Sync Baseline Cost to All Scenarios", on_click=sync_ordering_costs)
        
    st.markdown("Test the recommended baseline against custom strategies. Modify the review period to see the mathematically optimum target update instantly.")
    
    num_scenarios = st.slider("Select Number of Custom Scenarios to Compare:", min_value=1, max_value=5, value=2)
    
    scenarios_data = []
    s_cols = st.columns(num_scenarios)
    
    for i, col in enumerate(s_cols):
        with col:
            st.markdown(f"##### Scenario {i+1}")
            
            default_t = int(review_period + ((i+1) * 7)) 
            t_val = st.number_input(f"Review Period (Days)", value=default_t, min_value=1, step=1, key=f"t_{i}")
            
            u_pi = t_val + lead_time
            opt_target = (pr_avg_demand * u_pi) + (z_score * (pr_std_dev * np.sqrt(u_pi)))
            st.caption(f"✨ **Optimum Target:** {int(opt_target)} Units")
            target_val = st.number_input(f"Target Level (Units)", value=int(opt_target), step=50, key=f"target_{i}")
            
            if f"oc_key_{i}" not in st.session_state:
                st.session_state[f"oc_key_{i}"] = ordering_cost
                
            oc_val = st.number_input(f"Ordering Cost ($)", step=10.0, key=f"oc_key_{i}")
            
            scenarios_data.append({
                'name': f"Scenario {i+1}", 
                'T': t_val, 
                'Target': target_val, 
                'OrderCost': oc_val
            })

    # --- NumPy Optimized Simulation Engine ---
    np.random.seed(st.session_state.seed_counter)
    sim_days_pr = 365
    daily_demand_pr = np.clip(np.random.normal(pr_avg_demand, pr_std_dev, sim_days_pr), 0, None).round(0)

    def simulate_periodic_system_vectorized(demand_array, T, L, target, order_c, hold_c_daily):
        sim_days = len(demand_array)
        inventory_history = np.zeros(sim_days)
        receipts = np.zeros(sim_days + L + 1) 
        
        current_inv = target
        order_sizes = []
        units_fulfilled = 0
        
        for day in range(sim_days):
            current_inv += receipts[day]
            current_demand = demand_array[day]
            
            fulfilled = min(max(current_inv, 0), current_demand)
            current_inv -= fulfilled
            units_fulfilled += fulfilled
            
            inventory_history[day] = current_inv
            
            if day % T == 0:
                on_order = np.sum(receipts[day+1:day+L+1])
                inv_position = current_inv + on_order
                
                if inv_position < target:
                    order_qty = target - inv_position
                    receipts[day + L] += order_qty
                    order_sizes.append(order_qty)
                    
        holding_units_total = np.sum(np.maximum(inventory_history, 0))
        orders_placed = len(order_sizes)
        total_order_cost = orders_placed * order_c
        total_holding_cost = holding_units_total * hold_c_daily
        total_demand_sim = np.sum(demand_array)
        
        return {
            'history': inventory_history,
            'total_demand': total_demand_sim,
            'units_fulfilled': units_fulfilled,
            'lost_sales': total_demand_sim - units_fulfilled,
            'fill_rate': (units_fulfilled / total_demand_sim) * 100 if total_demand_sim > 0 else 0,
            'orders_placed': orders_placed,
            'min_order_size': np.min(order_sizes) if order_sizes else 0,
            'max_order_size': np.max(order_sizes) if order_sizes else 0,
            'avg_order_size': np.mean(order_sizes) if order_sizes else 0,
            'avg_inventory': holding_units_total / sim_days,
            'max_inventory': np.max(np.maximum(inventory_history, 0)),
            'min_inventory': np.min(inventory_history),
            'total_order_cost': total_order_cost,
            'total_holding_cost': total_holding_cost,
            'total_cost': total_order_cost + total_holding_cost
        }

    # Execute simulations 
    res_baseline = simulate_periodic_system_vectorized(daily_demand_pr, review_period, lead_time, recommended_target, ordering_cost, holding_cost_daily)
    
    scenario_results = []
    for s in scenarios_data:
        res = simulate_periodic_system_vectorized(daily_demand_pr, s['T'], lead_time, s['Target'], s['OrderCost'], holding_cost_daily)
        scenario_results.append(res)

    # --- 3. Logically Bifurcated Summary Tables ---
    st.divider()
    st.markdown("### 📊 Policy Comparison & KPI Summary")
    
    def fmt_usd(val): return f"${val:,.2f}"
    
    # 3A. Operational Health Matrix
    st.markdown("#### A. Operational & Capital Health Matrix")
    ops_data = {
        "Metric": [
            "Review Interval", 
            "Target Inventory Level", 
            "Fill Rate (%)", 
            "Lost Sales (Units)", 
            "Min Inventory Level (Depth)",
            "Avg Working Capital", 
            "Max Working Capital"
        ]
    }
    ops_data["Recommended Baseline"] = [
        f"{review_period} Days", f"{int(recommended_target)}", f"{res_baseline['fill_rate']:.2f}%", 
        f"{int(res_baseline['lost_sales'])}", f"{int(res_baseline['min_inventory'])}",
        fmt_usd(res_baseline['avg_inventory'] * unit_cost), fmt_usd(res_baseline['max_inventory'] * unit_cost)
    ]
    for idx, s in enumerate(scenarios_data):
        res = scenario_results[idx]
        ops_data[s['name']] = [
            f"{s['T']} Days", f"{int(s['Target'])}", f"{res['fill_rate']:.2f}%", 
            f"{int(res['lost_sales'])}", f"{int(res['min_inventory'])}",
            fmt_usd(res['avg_inventory'] * unit_cost), fmt_usd(res['max_inventory'] * unit_cost)
        ]
    st.dataframe(pd.DataFrame(ops_data), use_container_width=True, hide_index=True)

    # 3B. Order Dynamics Matrix
    st.markdown("#### B. Order Dynamics Matrix")
    order_data = {
        "Metric": ["Total No. of Orders", "Average Order Size", "Minimum Order Size", "Maximum Order Size"]
    }
    order_data["Recommended Baseline"] = [
        f"{res_baseline['orders_placed']}", f"{int(res_baseline['avg_order_size'])} Units", 
        f"{int(res_baseline['min_order_size'])} Units", f"{int(res_baseline['max_order_size'])} Units"
    ]
    for idx, s in enumerate(scenarios_data):
        res = scenario_results[idx]
        order_data[s['name']] = [
            f"{res['orders_placed']}", f"{int(res['avg_order_size'])} Units", 
            f"{int(res['min_order_size'])} Units", f"{int(res['max_order_size'])} Units"
        ]
    st.dataframe(pd.DataFrame(order_data), use_container_width=True, hide_index=True)

    # 3C. Financial Matrix
    st.markdown("#### C. Financial Projections Matrix")
    fin_data = {
        "Metric": ["Applied Ordering Cost ($/order)", "Total Ordering Cost", "Total Holding Cost", "Total System Cost"]
    }
    fin_data["Recommended Baseline"] = [
        fmt_usd(ordering_cost), fmt_usd(res_baseline['total_order_cost']), 
        fmt_usd(res_baseline['total_holding_cost']), fmt_usd(res_baseline['total_cost'])
    ]
    for idx, s in enumerate(scenarios_data):
        res = scenario_results[idx]
        fin_data[s['name']] = [
            fmt_usd(s['OrderCost']), fmt_usd(res['total_order_cost']), 
            fmt_usd(res['total_holding_cost']), fmt_usd(res['total_cost'])
        ]
    st.dataframe(pd.DataFrame(fin_data), use_container_width=True, hide_index=True)

    # --- 4. Visual Bifurcation & Trajectory ---
    chart_col1, chart_col2 = st.columns([1, 1])
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    with chart_col1:
        st.markdown("#### Cost Bifurcation Analysis")
        names = ["Baseline"] + [s['name'] for s in scenarios_data]
        order_costs = [res_baseline['total_order_cost']] + [r['total_order_cost'] for r in scenario_results]
        hold_costs = [res_baseline['total_holding_cost']] + [r['total_holding_cost'] for r in scenario_results]
        
        fig_cost = go.Figure(data=[
            go.Bar(name='Ordering Cost', x=names, y=order_costs, marker_color='#2ca02c'),
            go.Bar(name='Holding Cost', x=names, y=hold_costs, marker_color='#1f77b4')
        ])
        fig_cost.update_layout(
            barmode='stack', template="plotly_white", yaxis_title="Total Cost ($)",
            height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with chart_col2:
        st.markdown("#### Physical Inventory Trajectory")
        fig_comp = go.Figure()
        
        fig_comp.add_trace(go.Scatter(
            x=list(range(sim_days_pr)), y=res_baseline['history'], mode='lines', 
            name='Baseline', line=dict(color='#1f77b4', width=3)
        ))
        
        for idx, s in enumerate(scenarios_data):
            fig_comp.add_trace(go.Scatter(
                x=list(range(sim_days_pr)), y=scenario_results[idx]['history'], mode='lines', 
                name=s['name'], line=dict(color=colors[idx], width=1.5, dash='dot')
            ))
        
        fig_comp.add_hline(y=0, line_dash="solid", line_color="#333333", line_width=1)
        fig_comp.update_layout(
            template="plotly_white", xaxis_title="Simulation Day", yaxis_title="Units On Hand",
            height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # --- 5. Blocked Working Capital Chart ---
    st.write("<br>", unsafe_allow_html=True)
    st.markdown("#### 💰 Blocked Working Capital Trajectory")
    st.caption("Visualizes the daily capital tied up on the warehouse floor (ignores backorders).")
    
    fig_wc = go.Figure()
    
    # Baseline WC
    baseline_wc = np.maximum(res_baseline['history'], 0) * unit_cost
    fig_wc.add_trace(go.Scatter(
        x=list(range(sim_days_pr)), y=baseline_wc, mode='lines', 
        name='Baseline', line=dict(color='#1f77b4', width=3)
    ))
    
    # Scenarios WC
    for idx, s in enumerate(scenarios_data):
        scenario_wc = np.maximum(scenario_results[idx]['history'], 0) * unit_cost
        fig_wc.add_trace(go.Scatter(
            x=list(range(sim_days_pr)), y=scenario_wc, mode='lines', 
            name=s['name'], line=dict(color=colors[idx], width=1.5, dash='dot')
        ))
        
    fig_wc.update_layout(
        template="plotly_white", xaxis_title="Simulation Day", yaxis_title="Capital Blocked ($)",
        height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_wc, use_container_width=True)

    # --- 6. Collapsible Raw Data Logs ---
    with st.expander("📋 View Daily Simulation Log Tables"):
        st.markdown("Raw 365-day tracking for Physical Inventory levels side-by-side.")
        
        log_data = {
            "Day": range(1, sim_days_pr + 1),
            "Daily Demand": daily_demand_pr.astype(int),
            "Baseline Inv": res_baseline['history'].astype(int)
        }
        
        for idx, s in enumerate(scenarios_data):
            log_data[f"{s['name']} Inv"] = scenario_results[idx]['history'].astype(int)
            
        log_df = pd.DataFrame(log_data)
        
        def highlight_stockouts(val):
            color = '#ffcccc' if isinstance(val, (int, float)) and val < 0 else ''
            return f'background-color: {color}'
            
        st.dataframe(
            log_df.style.map(highlight_stockouts, subset=[c for c in log_df.columns if 'Inv' in c]), 
            use_container_width=True, hide_index=True
        )


# ==========================================
# TAB 6: INVENTORY SIMULATOR
# ==========================================
with tab6:
        st.header("Inventory Simulator")
        st.markdown(
            "Analyze your inventory data through a twin-lens framework. First, review a historical backtest audit "
            "to identify legacy profit leaks."
        )
        
        # --- 🚀 THE VECTORIZED SIMULATION ENGINE (DYNAMIC FIFO AGE BUCKETS) ---
        def fast_simulate_inventory(demand_arr, purchase_arr, opening_stock, lead_time, policy_type, param1, param2, age_bucket_edges=[30, 60, 90]):
            total_days = len(demand_arr)
            inv_levels = np.zeros(total_days)
            lost_sales = np.zeros(total_days)
            orders_placed = np.zeros(total_days)
            
            # Split positive receipts from negative returns to apply write-offs universally
            positive_receipts = np.where(purchase_arr > 0, purchase_arr, 0)
            negative_returns = np.where(purchase_arr < 0, np.abs(purchase_arr), 0)
            
            # FIFO Age Tracking Arrays
            avg_ages = np.zeros(total_days)
            max_ages = np.zeros(total_days)
            
            num_buckets = len(age_bucket_edges) + 1
            age_buckets = np.zeros((total_days, num_buckets)) 
            
            max_lt = int(lead_time)
            pipeline = np.zeros(total_days + max_lt + 1)
            
            if policy_type == "Actual":
                pipeline[:total_days] = positive_receipts
            else:
                warm_up = min(max_lt, total_days)
                pipeline[:warm_up] = positive_receipts[:warm_up]
                
            current_inv = opening_stock
            fifo_queue = [[0, opening_stock]] if opening_stock > 0 else []
            
            for i in range(total_days):
                demand = demand_arr[i]
                arriving = pipeline[i]
                daily_return = negative_returns[i]
                
                # 1. Process Positive Arriving Stock
                if arriving > 0:
                    fifo_queue.append([i, arriving])
                    current_inv += arriving
                    
                # 2. Mechanically Deduct Returns / Expiries / Write-offs
                if daily_return > 0:
                    current_inv = max(0, current_inv - daily_return)
                    # Deduct from FIFO queue to clear out the oldest physical stock
                    while daily_return > 0 and fifo_queue:
                        oldest_batch_qty = fifo_queue[0][1]
                        if oldest_batch_qty <= daily_return:
                            daily_return -= oldest_batch_qty
                            fifo_queue.pop(0)
                        else:
                            fifo_queue[0][1] -= daily_return
                            daily_return = 0
                            
                # 3. Fulfill Valid Customer Demand
                demand_left = demand
                while demand_left > 0 and fifo_queue:
                    oldest_batch_qty = fifo_queue[0][1]
                    if oldest_batch_qty <= demand_left:
                        demand_left -= oldest_batch_qty
                        fifo_queue.pop(0) 
                    else:
                        fifo_queue[0][1] -= demand_left
                        demand_left = 0 
                        
                if demand_left > 0:
                    lost_sales[i] = demand_left
                    current_inv = 0
                else:
                    current_inv -= demand
                    
                inv_levels[i] = current_inv
                
                # Calculate Age Metrics & Dynamic Buckets
                if fifo_queue:
                    total_qty = 0
                    sum_age_qty = 0
                    max_age_val = 0
                    
                    for item in fifo_queue:
                        arr_day, qty = item
                        age = i - arr_day
                        total_qty += qty
                        sum_age_qty += (age * qty)
                        
                        if age > max_age_val:
                            max_age_val = age
                            
                        # Dynamic bucket sorting
                        idx = 0
                        while idx < len(age_bucket_edges) and age > age_bucket_edges[idx]:
                            idx += 1
                        age_buckets[i, idx] += qty
                            
                    if total_qty > 0:
                        avg_ages[i] = sum_age_qty / total_qty
                        max_ages[i] = max_age_val
                
                if policy_type == "Continuous Review (Q, R)":
                    net_position = current_inv + np.sum(pipeline[i+1:])
                    if net_position <= param2:
                        pipeline[i + max_lt] += param1
                        orders_placed[i] = param1
                elif policy_type == "Periodic Review (P, T)":
                    if i % int(param1) == 0:
                        net_position = current_inv + np.sum(pipeline[i+1:])
                        order_qty = max(0, param2 - net_position)
                        if order_qty > 0:
                            pipeline[i + max_lt] += order_qty
                            orders_placed[i] = order_qty
                            
            return inv_levels, lost_sales, orders_placed, avg_ages, max_ages, age_buckets

        # --- STEP 1: INPUT PARAMETERS ---
        st.subheader("1. Parameters & Cost Drivers")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            item_unit_cost = st.number_input("Item Unit Cost ($/Unit)", min_value=0.01, value=25.00, step=1.00, key="unit_cost_global")
            holding_fixed_daily = st.number_input("Fixed Holding Cost ($/Unit/Day)", min_value=0.0, value=0.0, step=0.01, key="fixed_hold_global")
            
        with col2:
            holding_var_pct = st.number_input("Variable Holding Cost (% of Item Cost/year)", min_value=0.0, max_value=100.0, value=15.0, step=1.0, key="var_hold_global") / 100.0
            ordering_cost = st.number_input("Ordering Cost ($/order)", min_value=0.1, value=75.0, step=5.0, key="order_cost_global")
            
        with col3:
            lost_sales_penalty = st.number_input("Lost Sales Penalty ($/Unit Lost)", min_value=0.0, value=0.0, step=1.0, key="penalty_global")
            lead_time_days = st.number_input("Lead Time (Days)", min_value=1, value=14, step=1, key="lt_global")

        st.markdown("---")
        col_sys1, col_sys2 = st.columns([1, 2])
        with col_sys1:
            review_system = st.radio("Inventory Review System Strategy", ["Continuous Review (Q, R)", "Periodic Review (P, T)"], key="review_system_global")
        with col_sys2:
            service_level = st.slider("Target Service Level (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.5, key="service_level_global") / 100.0

        if review_system == "Periodic Review (P, T)":
            st.markdown("##### ⏳ Periodic Configuration")
            p_col1, _ = st.columns(2)
            with p_col1:
                user_p_days = st.number_input("Review Period Cycle (P in Days)", min_value=1, value=14, step=1, key="p_days_global")
        else:
            user_p_days = 1

        # --- STEP 2: MULTI-FORMAT DATA INGESTION ENGINE ---
        st.subheader("2. Upload Historical Invoices & Demand Data")
        uploaded_file = st.file_uploader(
            "Upload Inventory Ledger (Supports standard templates, raw ERP transactional logs, or stock card snapshots)", 
            type=["csv", "xlsx", "xls"], 
            key="uploader_global"
        )
        
        if uploaded_file is None:
            st.info("📥 Please upload your inventory ledger file (CSV or Excel) above to populate the suite modules.")
        else:
            detected_sheet_opening_stock = None
            data_loaded_successfully = False
            df_mapped = None
            
            try:
                if uploaded_file.name.endswith('.csv'):
                    raw_df = pd.read_csv(uploaded_file)
                else:
                    raw_df = pd.read_excel(uploaded_file)
                    
                raw_df.columns = raw_df.columns.str.strip()
                    
                if "Date" not in raw_df.columns:
                    st.error("❌ Missing required column: 'Date'.")
                else:
                    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
                    raw_df = raw_df.sort_values(by="Date").reset_index(drop=True)
                    
                    open_balance_headers = ["Opening Balance", "Opening", "Opening_Stock", "Opening Stock"]
                    for header in open_balance_headers:
                        if header in raw_df.columns:
                            detected_sheet_opening_stock = int(raw_df[header].iloc[0])
                            break
                    
                    if "Demand_Qty" in raw_df.columns and "Purchase_Qty" in raw_df.columns:
                        df_mapped = raw_df[["Date", "Demand_Qty", "Purchase_Qty"]].copy()
                    elif "Demand" in raw_df.columns and "Stock Received" in raw_df.columns:
                        df_mapped = pd.DataFrame({"Date": raw_df["Date"], "Demand_Qty": raw_df["Demand"], "Purchase_Qty": raw_df["Stock Received"]})
                    elif ("Receiving" in raw_df.columns) and any(col in raw_df.columns for col in ["Demand/Sales", "Demand", "Sales"]):
                        outbound_col = "Demand/Sales" if "Demand/Sales" in raw_df.columns else ("Demand" if "Demand" in raw_df.columns else "Sales")
                        df_mapped = pd.DataFrame({"Date": raw_df["Date"], "Demand_Qty": raw_df[outbound_col], "Purchase_Qty": raw_df["Receiving"]})
                    else:
                        st.error("❌ Column layout structure mismatch. Could not find Demand and Receiving columns.")

                    if df_mapped is not None:
                        df_mapped = df_mapped.groupby("Date").agg({"Demand_Qty": "sum", "Purchase_Qty": "sum"}).reset_index()
                        df_mapped = df_mapped.set_index("Date").resample("1D").asfreq()
                        df_mapped["Demand_Qty"] = df_mapped["Demand_Qty"].fillna(0.0)
                        df_mapped["Purchase_Qty"] = df_mapped["Purchase_Qty"].fillna(0.0)
                        df_mapped = df_mapped.reset_index()
                        data_loaded_successfully = True
                        
            except Exception as e:
                st.error(f"Error parsing file elements: {e}")

            # ONLY PROCEED IF DATA IS CLEANED
            if data_loaded_successfully and df_mapped is not None:
                full_df = df_mapped.copy() 
                
                absolute_min_date = full_df["Date"].min().date()
                absolute_max_date = full_df["Date"].max().date()
                
                file_state_key = f"last_file_{uploaded_file.name}_{uploaded_file.size}"
                
                if "current_file_token" not in st.session_state or st.session_state.current_file_token != file_state_key:
                    st.session_state.current_file_token = file_state_key
                    st.session_state.min_date_global = absolute_min_date
                    st.session_state.max_date_global = absolute_max_date
                    
                    avg_daily_full = full_df["Demand_Qty"].mean()
                    if detected_sheet_opening_stock is not None:
                        default_start = int(detected_sheet_opening_stock)
                    else:
                        default_start = int(1.25 * (avg_daily_full * lead_time_days))
                    
                    st.session_state.absolute_day1_stock = default_start
                    st.session_state.previous_start_date = absolute_min_date
                    st.session_state.previous_end_date = absolute_max_date
                    st.session_state.opening_stock_global = default_start
                    
                    st.session_state.start_date_key = absolute_min_date
                    st.session_state.end_date_key = absolute_max_date
                    
                    if "q_audit_suite" in st.session_state: del st.session_state.q_audit_suite
                    if "rop_audit_suite" in st.session_state: del st.session_state.rop_audit_suite

                st.divider()
                
                st.markdown("### 📅 3. Select Analysis Period")
                st.markdown("Filter historical data. The starting inventory will automatically mathematically roll forward to match your selected Start Date.")
                
                def reset_dates():
                    st.session_state.start_date_key = st.session_state.min_date_global
                    st.session_state.end_date_key = st.session_state.max_date_global
                    st.session_state.previous_start_date = st.session_state.min_date_global
                    st.session_state.previous_end_date = st.session_state.max_date_global
                    st.session_state.opening_stock_global = st.session_state.absolute_day1_stock

                col_date1, col_date2, col_date3 = st.columns([2, 2, 1])
                
                with col_date1:
                    start_date = st.date_input("Starting Date", min_value=absolute_min_date, max_value=absolute_max_date, key="start_date_key")
                with col_date2:
                    end_date = st.date_input("Ending Date", min_value=absolute_min_date, max_value=absolute_max_date, key="end_date_key")
                with col_date3:
                    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                    st.button("🔄 Reset", on_click=reset_dates, use_container_width=True)
                
                if "previous_start_date" not in st.session_state:
                    st.session_state.previous_start_date = absolute_min_date
                if "previous_end_date" not in st.session_state:
                    st.session_state.previous_end_date = absolute_max_date

                date_changed = (start_date != st.session_state.previous_start_date) or (end_date != st.session_state.previous_end_date)

                if date_changed:
                    if "q_audit_suite" in st.session_state: del st.session_state.q_audit_suite
                    if "rop_audit_suite" in st.session_state: del st.session_state.rop_audit_suite
                    
                    if start_date != st.session_state.previous_start_date:
                        temp_balance = st.session_state.absolute_day1_stock
                        pre_period_df = full_df[full_df["Date"].dt.date < start_date]
                        
                        pre_d_arr = pre_period_df["Demand_Qty"].values
                        pre_p_arr = pre_period_df["Purchase_Qty"].values
                        for idx in range(len(pre_d_arr)):
                            temp_balance = max(0, temp_balance + pre_p_arr[idx] - pre_d_arr[idx])
                        
                        st.session_state.opening_stock_global = int(temp_balance)
                    
                    st.session_state.previous_start_date = start_date
                    st.session_state.previous_end_date = end_date

                if start_date > end_date:
                    st.error("⚠️ The Starting Date must be before or equal to the Ending Date. Please adjust your selection.")
                else:
                    df = full_df[(full_df["Date"].dt.date >= start_date) & (full_df["Date"].dt.date <= end_date)].reset_index(drop=True)
                    
                    if df.empty:
                        st.warning("No data available for the selected date range. Please widen your selection.")
                    else:
                        demand_arr_main = df["Demand_Qty"].values
                        purchase_arr_main = df["Purchase_Qty"].values
                        
                        # Filter for actual positive purchase orders to calculate correct baseline KPIs
                        actual_orders_placed = np.count_nonzero(purchase_arr_main[purchase_arr_main > 0])
                        actual_total_units_purchased = purchase_arr_main[purchase_arr_main > 0].sum()
                        total_demand = demand_arr_main.sum()
                        
                        avg_daily_demand_calc = demand_arr_main.mean()
                        std_daily_demand = demand_arr_main.std() if len(df) > 1 else 0
                        cov = std_daily_demand / max(0.1, avg_daily_demand_calc)

                    with st.expander("📈 View Historical Demand Trend & Growth Timeline", expanded=False):
                        rolling_days = st.slider("Select Rolling Average Window (Days)", min_value=1, max_value=90, value=15, step=1)
                        df[f"Rolling_Avg"] = df["Demand_Qty"].rolling(window=rolling_days, min_periods=1).mean()
                        
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        
                        if len(df) >= rolling_days * 2:
                            current_window_sum = df["Demand_Qty"].iloc[-rolling_days:].sum()
                            previous_window_sum = df["Demand_Qty"].iloc[-(rolling_days*2):-rolling_days].sum()
                            if previous_window_sum > 0:
                                trend_pct = ((current_window_sum - previous_window_sum) / previous_window_sum) * 100
                            else:
                                trend_pct = 100.0 if current_window_sum > 0 else 0.0
                            stat_col1.metric(label=f"Trend (Last {rolling_days} Days)", value=f"{current_window_sum:,.0f} units", delta=f"{trend_pct:+.1f}% Growth", delta_color="normal")
                        elif len(df) >= rolling_days:
                            current_window_sum = df["Demand_Qty"].iloc[-rolling_days:].sum()
                            stat_col1.metric(label=f"Trend (Last {rolling_days} Days)", value=f"{current_window_sum:,.0f} units", delta="Widen dates for trend", delta_color="off")
                        else:
                            stat_col1.metric(label="Trend", value="N/A", delta="Insufficient Data", delta_color="off")
                            
                        stat_col2.metric("Average Daily Demand", f"{avg_daily_demand_calc:.1f} units")
                        stat_col3.metric("Standard Deviation", f"{std_daily_demand:.1f} units")
                        stat_col4.metric("Volatility (CoV)", f"{cov:.2f}")
                        st.markdown("---")

                        demand_fig = go.Figure()
                        demand_fig.add_trace(go.Scatter(x=df["Date"], y=df["Demand_Qty"], mode='lines', name='Raw Daily Demand', line=dict(color='#B0C4DE', width=1.5), opacity=0.6))
                        demand_fig.add_trace(go.Scatter(x=df["Date"], y=df["Rolling_Avg"], mode='lines', name=f'{rolling_days}-Day Moving Avg', line=dict(color='#3d5a80', width=3)))
                        demand_fig.update_layout(template="plotly_white", yaxis_title="Units", xaxis_title="Date", margin=dict(t=20, b=20), height=350, legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
                        st.plotly_chart(demand_fig, use_container_width=True)

                    st.markdown("---")
                    st.markdown("##### 📦 Initial Warehouse Capital Balance")
                    opening_stock_override = st.number_input("Starting Balance for Selected Period", min_value=0, step=10, key="opening_stock_global", help="This value automatically updates mathematically based on the start date you select above, but you can override it manually.")

                    with st.expander("📋 View Complete Running Balance Table Snapshots", expanded=False):
                        st.markdown("An interactive historical stock card ledger driven directly by your initial opening stock parameter above.")
                        cl_open_list, cl_close_list, t_bal = np.zeros(len(df)), np.zeros(len(df)), opening_stock_override
                        
                        for i_run in range(len(df)):
                            cl_open_list[i_run] = t_bal
                            t_bal = max(0, t_bal + purchase_arr_main[i_run] - demand_arr_main[i_run])
                            cl_close_list[i_run] = t_bal
                            
                        full_stock_card_df = pd.DataFrame({
                            "Timeline Date": df["Date"].dt.strftime('%Y-%m-%d'),
                            "Opening Balance": cl_open_list.astype(int),
                            "Cleaned Demand Volume (Units)": demand_arr_main.astype(int),
                            "Consolidated Stock Received (Units)": purchase_arr_main.astype(int),
                            "Closing Balance": cl_close_list.astype(int)
                        })
                        
                        st.dataframe(full_stock_card_df, use_container_width=True, hide_index=True, column_config={"Opening Balance": st.column_config.NumberColumn(format="%d"), "Cleaned Demand Volume (Units)": st.column_config.NumberColumn(format="%d"), "Consolidated Stock Received (Units)": st.column_config.NumberColumn(format="%d"), "Closing Balance": st.column_config.NumberColumn(format="%d")})

                    st.subheader("4. Statistical Risk & Distribution Engines")
                    
                    annual_demand = avg_daily_demand_calc * 365
                    annual_fixed_holding_per_unit = holding_fixed_daily * 365
                    unit_holding_cost = annual_fixed_holding_per_unit + (item_unit_cost * holding_var_pct)
                    
                    risk_horizon_days = lead_time_days if review_system == "Continuous Review (Q, R)" else (user_p_days + lead_time_days)
                    rolling_risk_demand = df["Demand_Qty"].rolling(window=int(risk_horizon_days)).sum().dropna().values
                    risk_mean = np.mean(rolling_risk_demand) if len(rolling_risk_demand) > 0 else 0
                    risk_std = np.std(rolling_risk_demand) if len(rolling_risk_demand) > 0 else 0

                    if len(rolling_risk_demand) > 0:
                        if np.max(rolling_risk_demand) <= 0:
                            best_fit_name = "Zero Demand Base"
                            raw_target_level = 0.0
                        else:
                            empirical_rop_raw = np.percentile(rolling_risk_demand, service_level * 100)
                            safe_demand = np.where(rolling_risk_demand <= 0, 1e-5, rolling_risk_demand)
                            
                            log_params = stats.lognorm.fit(safe_demand, floc=0)
                            gam_params = stats.gamma.fit(safe_demand, floc=0)

                            counts, bins = np.histogram(rolling_risk_demand, bins=20, density=True)
                            bin_centers = (bins[:-1] + bins[1:]) / 2
                            
                            rss_norm = np.sum((counts - stats.norm.pdf(bin_centers, loc=risk_mean, scale=risk_std)) ** 2)
                            rss_log = np.sum((counts - stats.lognorm.pdf(bin_centers, *log_params)) ** 2)
                            rss_gam = np.sum((counts - stats.gamma.pdf(bin_centers, *gam_params)) ** 2)

                            if cov > 0.75:
                                best_fit_name = "Empirical (Data-Driven)"
                                raw_target_level = empirical_rop_raw
                            else:
                                errors = {"Normal": rss_norm, "Log-Normal": rss_log, "Gamma": rss_gam}
                                best_fit_name = min(errors, key=errors.get)
                                if best_fit_name == "Normal":
                                    raw_target_level = stats.norm.ppf(service_level, loc=risk_mean, scale=risk_std)
                                elif best_fit_name == "Log-Normal":
                                    raw_target_level = stats.lognorm.ppf(service_level, *log_params)
                                else:
                                    raw_target_level = stats.gamma.ppf(service_level, *gam_params)
                    else:
                        best_fit_name = "Default (Insufficient Data)"
                        raw_target_level = avg_daily_demand_calc * risk_horizon_days
                        
                    raw_optimal_q = np.sqrt((2 * annual_demand * ordering_cost) / max(0.01, unit_holding_cost))

                    if "q_audit_suite" not in st.session_state:
                        st.session_state.q_audit_suite = max(1, int(raw_optimal_q)) if review_system == "Continuous Review (Q, R)" else int(avg_daily_demand_calc * user_p_days)
                    if "rop_audit_suite" not in st.session_state:
                        st.session_state.rop_audit_suite = max(0, int(raw_target_level))

                    with st.expander("📊 View Cleaned Demand Distribution & Best-Fit Curve Metrics", expanded=False):
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        stat_col1.metric("Average Daily Demand", f"{avg_daily_demand_calc:.2f} units")
                        stat_col2.metric("Coefficient of Variation (CV)", f"{cov:.2f}")
                        stat_col3.metric("Engine Selection", f"✨ {best_fit_name}")
                        st.markdown("---")
                        hist_fig = go.Figure()
                        hist_fig.add_trace(go.Histogram(x=df["Demand_Qty"], name="Historical Days", marker_color='#1F77B4', opacity=0.6))
                        hist_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Demand Quantity (Units / Day)", yaxis_title="Frequency", margin=dict(l=40, r=40, t=10, b=40), height=300)
                        st.plotly_chart(hist_fig, use_container_width=True)

                    st.markdown("---")
                    st.subheader("5. 🌪️ Erratic Demand & Empirical ROP Profiler")
                    
                    with st.expander("Open Rolling Window & Empirical Profiler", expanded=False):
                        st.markdown(
                            "Standard safety stock math assumes demand follows a predictable bell curve. For erratic or lumpy demand, "
                            "that assumption breaks down. This tool mechanically profiles your exact historical risk by analyzing every "
                            "rolling vulnerability window in your dataset."
                        )
                        
                        tab_emp_cont, tab_emp_per = st.tabs(["📉 Continuous Review (ROP)", "⏳ Periodic Review (Target Level)"])
                        
                        with tab_emp_cont:
                            st.markdown("##### Continuous Review Risk Profiler")
                            col_emp1, col_emp2, col_emp3 = st.columns(3)
                            with col_emp1:
                                emp_lt_window = st.number_input("Lead Time (Days)", min_value=1, value=int(lead_time_days), step=1, key="emp_lt_window")
                            with col_emp2:
                                emp_test_rop = st.number_input("Test Reorder Point (ROP)", min_value=0, value=int(raw_target_level), step=10, key="emp_test_rop")
                            with col_emp3:
                                emp_target_sl = st.number_input("Target Service Level (%)", min_value=1.0, max_value=99.9, value=95.0, step=0.5, key="emp_target_sl")

                            rolling_demand_c = df["Demand_Qty"].rolling(window=emp_lt_window).sum().dropna()

                            if len(rolling_demand_c) > 0:
                                windows_below_rop = np.sum(rolling_demand_c <= emp_test_rop)
                                total_windows_c = len(rolling_demand_c)
                                achieved_sl_c = (windows_below_rop / total_windows_c) * 100
                                required_rop = np.percentile(rolling_demand_c, emp_target_sl)

                                res_col1, res_col2 = st.columns(2)
                                with res_col1:
                                    st.info(f"**Testing ROP of {emp_test_rop:,}:**\n\nOut of {total_windows_c:,} historical {emp_lt_window}-day windows, the total demand was successfully covered by {emp_test_rop:,} units exactly **{windows_below_rop:,} times**. This yields an empirical service level of **{achieved_sl_c:.1f}%**.")
                                with res_col2:
                                    st.success(f"**Targeting {emp_target_sl}% Service Level:**\n\nTo mechanically guarantee that you don't stock out in {emp_target_sl}% of all historical {emp_lt_window}-day scenarios, your ROP must be set to the empirical percentile: **{int(required_rop):,} units**.")

                                emp_fig_c = go.Figure()
                                emp_fig_c.add_trace(go.Histogram(x=rolling_demand_c, nbinsx=40, marker_color='#B0C4DE', name=f"Historical {emp_lt_window}-Day Windows"))
                                emp_fig_c.add_vline(x=emp_test_rop, line_width=2, line_dash="dash", line_color="#FF4B4B", annotation_text=f"Tested ROP ({emp_test_rop})", annotation_position="top right")
                                emp_fig_c.add_vline(x=required_rop, line_width=2, line_dash="dash", line_color="#1F77B4", annotation_text=f"Target ROP ({int(required_rop)})", annotation_position="top left")

                                emp_fig_c.update_layout(title=f"Actual Demand Distribution Across All {emp_lt_window}-Day Windows", xaxis_title=f"Total Units Demanded in a {emp_lt_window}-Day Window", yaxis_title="Frequency", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(t=40, b=40))
                                st.plotly_chart(emp_fig_c, use_container_width=True)
                            else:
                                st.warning(f"⚠️ Not enough data to calculate rolling windows for a {emp_lt_window}-day lead time.")

                        with tab_emp_per:
                            st.markdown("##### Periodic Review Risk Profiler")
                            default_p_val = int(user_p_days) if user_p_days > 1 else 14
                            default_t_val = int(raw_target_level + (avg_daily_demand_calc * default_p_val))
                            
                            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                            with p_col1:
                                emp_p_days = st.number_input("Review Period (P)", min_value=1, value=default_p_val, step=1, key="emp_p_days")
                            with p_col2:
                                emp_p_lt = st.number_input("Lead Time (Days)", min_value=1, value=int(lead_time_days), step=1, key="emp_p_lt")
                            with p_col3:
                                emp_test_t = st.number_input("Test Target Level (T)", min_value=0, value=default_t_val, step=10, key="emp_test_t")
                            with p_col4:
                                emp_target_sl_p = st.number_input("Target Service Level (%)", min_value=1.0, max_value=99.9, value=95.0, step=0.5, key="emp_target_sl_p")

                            risk_window_days = emp_p_days + emp_p_lt
                            rolling_demand_p = df["Demand_Qty"].rolling(window=risk_window_days).sum().dropna()

                            if len(rolling_demand_p) > 0:
                                windows_below_t = np.sum(rolling_demand_p <= emp_test_t)
                                total_windows_p = len(rolling_demand_p)
                                achieved_sl_p = (windows_below_t / total_windows_p) * 100
                                required_t = np.percentile(rolling_demand_p, emp_target_sl_p)

                                res_col3, res_col4 = st.columns(2)
                                with res_col3:
                                    st.info(f"**Testing Target (T) of {emp_test_t:,}:**\n\nOut of {total_windows_p:,} historical {risk_window_days}-day windows (P+L), demand was successfully covered by {emp_test_t:,} units exactly **{windows_below_t:,} times**. Empirical SL: **{achieved_sl_p:.1f}%**.")
                                with res_col4:
                                    st.success(f"**Targeting {emp_target_sl_p}% Service Level:**\n\nTo mechanically guarantee that you don't stock out in {emp_target_sl_p}% of all historical {risk_window_days}-day scenarios, your Target Level must be: **{int(required_t):,} units**.")

                                emp_fig_p = go.Figure()
                                emp_fig_p.add_trace(go.Histogram(x=rolling_demand_p, nbinsx=40, marker_color='#B0C4DE', name=f"Historical {risk_window_days}-Day Windows"))
                                emp_fig_p.add_vline(x=emp_test_t, line_width=2, line_dash="dash", line_color="#FF4B4B", annotation_text=f"Tested Target ({emp_test_t})", annotation_position="top right")
                                emp_fig_p.add_vline(x=required_t, line_width=2, line_dash="dash", line_color="#1F77B4", annotation_text=f"Required Target ({int(required_t)})", annotation_position="top left")

                                emp_fig_p.update_layout(title=f"Actual Demand Distribution Across All {risk_window_days}-Day (P+L) Windows", xaxis_title=f"Total Units Demanded in a {risk_window_days}-Day Window", yaxis_title="Frequency", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(t=40, b=40))
                                st.plotly_chart(emp_fig_p, use_container_width=True)
                            else:
                                st.warning(f"⚠️ Not enough data to calculate rolling windows for a {risk_window_days}-day (P+L) window.")

                    st.markdown("---")
                    st.subheader("6. Policy Control & Aging Parameters")
                    
                    adjust_col1, adjust_col2 = st.columns(2)
                    with adjust_col1:
                        if review_system == "Continuous Review (Q, R)":
                            final_q = st.number_input("Target Order Quantity (Q)", min_value=1, step=10, key="q_audit_suite")
                        else:
                            cycle_demand_baseline = max(1, int(avg_daily_demand_calc * user_p_days))
                            final_q = st.number_input("Average Target Batch Size (Q)", min_value=1, value=cycle_demand_baseline, step=10, disabled=True, key="q_audit_suite_disabled")
                    with adjust_col2:
                        final_buffer_target = st.number_input("Reorder Point (ROP) / Target Level (T)", min_value=0, step=10, key="rop_audit_suite")

                    if review_system == "Continuous Review (Q, R)":
                        st.info(f"🎯 **Engine-Calculated Benchmarks ({best_fit_name}):** Optimal Order Quantity (EOQ): **{int(raw_optimal_q):,}** units | Recommended Reorder Point (ROP): **{int(raw_target_level):,}** units.")
                    else:
                        st.info(f"🎯 **Engine-Calculated Benchmarks ({best_fit_name}):** Expected Cycle Batch Size: **{int(avg_daily_demand_calc * user_p_days):,}** units | Recommended Max Order Up-To Level (T): **{int(raw_target_level):,}** units.")

                    optimal_p_days = max(1, int((final_q / max(0.1, avg_daily_demand_calc)))) if review_system == "Continuous Review (Q, R)" else int(user_p_days)

                    # --- UI CONTAINERS TO ENFORCE RENDERING ORDER ---
                    header_container = st.container()
                    kpi_container = st.container()
                    matrix_container = st.container()
                    timeline_container = st.container()
                    age_input_container = st.container()
                    age_profile_container = st.container()
                    drilldown_container = st.container()

                    # 1. Capture the age bucket definitions (renders physically in the 5th container, evaluates here logically)
                    with age_input_container:
                        st.markdown("---")
                        st.markdown("### 🕰️ Inventory Age Profile (FIFO Stacked)")
                        st.markdown("Visualize what fraction of your total inventory sitting in the warehouse belongs to distinct aging brackets over time.")
                        bucket_input = st.text_input("Define custom inventory age thresholds in days (comma-separated, e.g., '15, 30, 45, 60, 90')", value="30, 60, 90")
                        
                    try:
                        custom_edges = sorted(list(set([int(x.strip()) for x in bucket_input.split(',') if x.strip().isdigit()])))
                        if not custom_edges: custom_edges = [30, 60, 90]
                    except:
                        custom_edges = [30, 60, 90]

                    bucket_labels = []
                    prev_edge = 0
                    for edge in custom_edges:
                        bucket_labels.append(f"{prev_edge}-{edge} Days")
                        prev_edge = edge + 1
                    bucket_labels.append(f"{prev_edge}+ Days")

                    num_buckets = len(bucket_labels)
                    
                    if num_buckets == 1:
                        bucket_colors = ["#8EC9FF"]
                    else:
                        bucket_colors = px.colors.sample_colorscale(
                            [
                                [0.0, "#8EC9FF"],
                                [0.4, "#1E88E5"],
                                [0.7, "#FFA726"],
                                [1.0, "#E53935"]
                            ],
                            [i/(num_buckets-1) for i in range(num_buckets)]
                        )

                    # 2. RUN SIMULATIONS (Hidden from UI, populates downstream components)
                    with header_container:
                        st.markdown("---")
                        st.header("📊 Section A: Historical Backtest Audit")
                        st.markdown("This analysis compares your **Historical Actuals** against our **Recommended Optimized Policy** under identical historical demand constraints to reveal operational friction.")

                    # Actuals
                    inv_levels_act, lost_sales_act_arr, orders_placed_act_arr, avg_age_act, max_age_act, buckets_act = fast_simulate_inventory(
                        demand_arr_main, purchase_arr_main, opening_stock_override, lead_time_days, "Actual", 0, 0, custom_edges
                    )
                    
                    # Optimized
                    inv_levels_opt, lost_sales_opt_arr, orders_placed_opt_arr, avg_age_opt, max_age_opt, buckets_opt = fast_simulate_inventory(
                        demand_arr_main, purchase_arr_main, opening_stock_override, lead_time_days, review_system, 
                        optimal_p_days if review_system != "Continuous Review (Q, R)" else final_q, final_buffer_target, custom_edges
                    )

                    # --- CALCULATE METRICS ---
                    lost_sales_qty_act = lost_sales_act_arr.sum()
                    stockout_days_act = np.count_nonzero(lost_sales_act_arr)
                    zero_stock_days_act = np.count_nonzero(inv_levels_act == 0)
                    
                    actual_max_inventory = np.max(inv_levels_act)
                    actual_min_inventory = np.min(inv_levels_act)
                    actual_avg_inventory = np.mean(inv_levels_act)
                    actual_fill_rate = max(0.0, 1.0 - (lost_sales_qty_act / max(1, total_demand)))
                    actual_cycle_time = 365 / actual_orders_placed if actual_orders_placed > 0 else 365.0
                    actual_avg_order_size = actual_total_units_purchased / actual_orders_placed if actual_orders_placed > 0 else 0.0

                    actual_total_ordering_cost = actual_orders_placed * ordering_cost
                    actual_total_holding_cost = actual_avg_inventory * unit_holding_cost
                    actual_lost_sales_financial = lost_sales_qty_act * lost_sales_penalty
                    actual_total_cost = actual_total_ordering_cost + actual_total_holding_cost + actual_lost_sales_financial
                    actual_overall_avg_age = np.mean(avg_age_act)
                    actual_overall_max_age = np.max(max_age_act)

                    lost_sales_qty_opt = lost_sales_opt_arr.sum()
                    stockout_days_opt = np.count_nonzero(lost_sales_opt_arr)
                    zero_stock_days_opt = np.count_nonzero(inv_levels_opt == 0)
                    opt_orders_placed = np.count_nonzero(orders_placed_opt_arr)
                    policy_total_units_ordered = orders_placed_opt_arr.sum()

                    simmed_avg_opt_inv = np.mean(inv_levels_opt)
                    simmed_max_opt_inv = np.max(inv_levels_opt)
                    simmed_min_inventory = np.min(inv_levels_opt)
                    simmed_opt_fill_rate = max(0.0, 1.0 - (lost_sales_qty_opt / max(1, total_demand)))
                    policy_cycle_time = 365 / opt_orders_placed if opt_orders_placed > 0 else 365.0
                    policy_avg_order_size = policy_total_units_ordered / opt_orders_placed if opt_orders_placed > 0 else 0.0

                    optimal_ordering_cost = opt_orders_placed * ordering_cost
                    optimal_holding_cost = simmed_avg_opt_inv * unit_holding_cost
                    optimal_lost_sales_financial = lost_sales_qty_opt * lost_sales_penalty
                    optimal_total_cost = optimal_ordering_cost + optimal_holding_cost + optimal_lost_sales_financial
                    opt_overall_avg_age = np.mean(avg_age_opt)
                    opt_overall_max_age = np.max(max_age_opt)
                    
                    act_max_wc = actual_max_inventory * item_unit_cost
                    act_avg_wc = actual_avg_inventory * item_unit_cost
                    act_min_wc = actual_min_inventory * item_unit_cost
                    
                    opt_max_wc = simmed_max_opt_inv * item_unit_cost
                    opt_avg_wc = simmed_avg_opt_inv * item_unit_cost
                    opt_min_wc = simmed_min_inventory * item_unit_cost

                    true_net_benefit = actual_total_cost - optimal_total_cost

                    # 3. POPULATE UI CONTAINERS
                    with kpi_container:
                        if true_net_benefit > 0:
                            st.success(f"### 🎯 The Efficiency Opportunity\nBy shifting to the recommended optimized policy, you would have recovered **${true_net_benefit:,.2f}** over this historical period.")
                        else:
                            st.error(f"⚠️ **Operational Margin Deficit Risk:** This setup increases operational overhead by **${abs(true_net_benefit):,.2f} / year** compared to actuals.")

                        st.markdown("### 🏆 Executive Summary: Value Realization")
                        cash_released = act_avg_wc - opt_avg_wc

                        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                        with kpi_col1: st.metric(label="Total Cost Saving", value=f"${true_net_benefit:,.0f}")
                        with kpi_col2: st.metric(label="Optimized Fill Rate", value=f"{simmed_opt_fill_rate * 100:.1f}%")
                        with kpi_col3:
                            st.metric(label="Avg Working Capital (Opt)", value=f"${opt_avg_wc:,.0f}")
                            st.markdown(f"<div style='margin-top: -15px; font-size: 0.85rem; color: gray;'>Historical: ${act_avg_wc:,.0f}</div>", unsafe_allow_html=True)
                        with kpi_col4:
                            release_label = "Cash Released" if cash_released >= 0 else "Capital Added (Tied Up)"
                            # st.metric(label=release_label, value=f"${abs(cash_released):,.0f}")
                            cash_released_pct = (cash_released / act_avg_wc) * 100 if act_avg_wc > 0 else 0.0
                            st.metric(label=release_label, value=f"${abs(cash_released):,.0f}", delta=f"{cash_released_pct:+.1f}%")
                        st.markdown("---")

                    with matrix_container:
                        def render_clustered_matrix(title, metrics, act_vals, pol_vals, formats):
                            st.markdown(f"#### {title}")
                            abs_var = [a - p for a, p in zip(act_vals, pol_vals)]
                            pct_var = []
                            for a, p in zip(act_vals, pol_vals):
                                if a == 0: pct_var.append(0.0)
                                else: pct_var.append(((a - p) / a) * 100)
                                
                            m_df = pd.DataFrame({"Operational Attribute Pillar": metrics})
                            for idx in range(len(metrics)):
                                fmt = formats[idx]
                                if fmt == "currency":
                                    m_df.at[idx, "Historical Actuals"] = f"${act_vals[idx]:,.2f}"
                                    m_df.at[idx, "Optimized Policy"] = f"${pol_vals[idx]:,.2f}"
                                    m_df.at[idx, "Net Delta Variance"] = f"${abs_var[idx]:,.2f}" if abs_var[idx] >= 0 else f"-${abs(abs_var[idx]):,.2f}"
                                    m_df.at[idx, "% Impact Efficiency"] = f"{pct_var[idx]:+.1f}%"
                                elif fmt == "pct":
                                    m_df.at[idx, "Historical Actuals"] = f"{act_vals[idx]:.1f}%"
                                    m_df.at[idx, "Optimized Policy"] = f"{pol_vals[idx]:.1f}%"
                                    m_df.at[idx, "Net Delta Variance"] = f"{abs_var[idx]:+.1f}% pts"
                                    m_df.at[idx, "% Impact Efficiency"] = f"{pol_vals[idx] - act_vals[idx]:+.1f}% pts"
                                elif fmt == "days":
                                    m_df.at[idx, "Historical Actuals"] = f"{act_vals[idx]:,.1f} days"
                                    m_df.at[idx, "Optimized Policy"] = f"{pol_vals[idx]:,.1f} days"
                                    m_df.at[idx, "Net Delta Variance"] = f"{abs_var[idx]:+,.1f} days"
                                    m_df.at[idx, "% Impact Efficiency"] = f"{pct_var[idx]:+.1f}%"
                                else:
                                    m_df.at[idx, "Historical Actuals"] = f"{int(act_vals[idx]):,}"
                                    m_df.at[idx, "Optimized Policy"] = f"{int(pol_vals[idx]):,}"
                                    m_df.at[idx, "Net Delta Variance"] = f"{int(abs_var[idx]):+1,}"
                                    m_df.at[idx, "% Impact Efficiency"] = f"{pct_var[idx]:+.1f}%"

                            def apply_matrix_styles(x):
                                colors = pd.DataFrame('', index=x.index, columns=x.columns)
                                fav = 'background-color: #1A3E2B; color: #81C784; font-weight: bold;'
                                unfav = 'background-color: #3E1A1A; color: #E57373;'
                                for i, metric in enumerate(metrics):
                                    v = abs_var[i]
                                    if title == "1. Financial Breakdown Matrix" or title == "3. Working Capital Release Matrix":
                                        if v > 0: colors.iloc[i, 3:] = fav
                                        elif v < 0: colors.iloc[i, 3:] = unfav
                                    elif title == "4. Stockout Risk & Vulnerability Matrix":
                                        if "Fill Rate" in metric:
                                            if pol_vals[i] > act_vals[i]: colors.iloc[i, 3:] = fav
                                            elif pol_vals[i] < act_vals[i]: colors.iloc[i, 3:] = unfav
                                        else:
                                            if v > 0: colors.iloc[i, 3:] = fav
                                            elif v < 0: colors.iloc[i, 3:] = unfav
                                    elif title == "5. Inventory Age & Freshness Matrix (FIFO)":
                                        if v > 0: colors.iloc[i, 3:] = fav   
                                        elif v < 0: colors.iloc[i, 3:] = unfav
                                return colors
                            st.dataframe(m_df.style.apply(apply_matrix_styles, axis=None), use_container_width=True, hide_index=True)

                        render_clustered_matrix("1. Financial Breakdown Matrix", ["Annual Ordering Fees ($)", "Annual Storage Carrying Cost ($)", "Financial Penalty from Stockouts ($)", "Total Policy Operating Cost ($)"], [actual_total_ordering_cost, actual_total_holding_cost, actual_lost_sales_financial, actual_total_cost], [optimal_ordering_cost, optimal_holding_cost, optimal_lost_sales_financial, optimal_total_cost], ["currency", "currency", "currency", "currency"])
                        render_clustered_matrix("2. Logistical Operations Footprint Matrix", ["Average Volume Kept On-Hand", "Maximum Storage Spike Level", "Total Orders Dispatched", "Average Logistics Cycle Time", "Average Order Shipment Size"], [actual_avg_inventory, actual_max_inventory, actual_orders_placed, actual_cycle_time, actual_avg_order_size], [simmed_avg_opt_inv, simmed_max_opt_inv, opt_orders_placed, policy_cycle_time, policy_avg_order_size], ["units", "units", "count", "days", "units"])
                        render_clustered_matrix("3. Working Capital Release Matrix", ["Peak Working Capital Tied Up ($)", "Average Working Capital Tied Up ($)", "Minimum Base Working Capital ($)"], [act_max_wc, act_avg_wc, act_min_wc], [opt_max_wc, opt_avg_wc, opt_min_wc], ["currency", "currency", "currency"])
                        render_clustered_matrix("4. Stockout Risk & Vulnerability Matrix", ["Absolute Minimum Buffer Stock", "Stockout Events (Unfulfilled Days)", "Total Unfulfilled Deficit Volume", "Days with Absolute Zero Closing Stock", "Achieved Order Fill Rate (%)"], [actual_min_inventory, stockout_days_act, lost_sales_qty_act, zero_stock_days_act, actual_fill_rate * 100], [simmed_min_inventory, stockout_days_opt, lost_sales_qty_opt, zero_stock_days_opt, simmed_opt_fill_rate * 100], ["units", "count", "units", "count", "pct"])
                        render_clustered_matrix("5. Inventory Age & Freshness Matrix (FIFO)", ["Overall Average Inventory Age (Days)", "Maximum Peak Inventory Age (Days)"], [actual_overall_avg_age, actual_overall_max_age], [opt_overall_avg_age, opt_overall_max_age], ["days", "days"])

                    with timeline_container:
                        st.markdown("---")
                        st.markdown("### 📈 Tactical Operations Timeline Visualizations")
                        timeline_fig = go.Figure()
                        timeline_fig.add_trace(go.Scatter(x=df["Date"], y=inv_levels_act, name="Historical Actuals (Ledger)", line=dict(color='#B0C4DE', width=2), fill='tozeroy', fillcolor='rgba(176, 196, 222, 0.15)'))
                        timeline_fig.add_trace(go.Scatter(x=df["Date"], y=inv_levels_opt, name=f"Recommended Optimized Policy ({best_fit_name.split(' ')[0]})", line=dict(color='#1F77B4', width=2.5)))
                        timeline_fig.add_trace(go.Scatter(x=df["Date"], y=[max(0, raw_target_level - risk_mean)] * len(df), name="Calculated Safety Stock Floor", line=dict(color='#FF4B4B', width=1.5, dash='dot')))
                        timeline_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Timeline Date", yaxis_title="On-Hand Inventory (Units)", height=350, legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
                        st.plotly_chart(timeline_fig, use_container_width=True)

                    with age_profile_container:
                        age_tab1, age_tab2 = st.tabs(["📉 Historical Actuals Age Profile", "⚙️ Optimized Policy Age Profile"])
                        
                        with age_tab1:
                            fig_act_age = go.Figure()
                            for j in range(len(bucket_labels)):
                                fig_act_age.add_trace(go.Scatter(
                                    x=df["Date"], y=buckets_act[:, j], name=bucket_labels[j],
                                    mode='lines', stackgroup='one', line=dict(width=0.5, color=bucket_colors[j])
                                ))
                            fig_act_age.update_layout(template="plotly_white", yaxis_title="Units In Stock", xaxis_title="Date", height=350)
                            st.plotly_chart(fig_act_age, use_container_width=True)
                            
                        with age_tab2:
                            fig_opt_age = go.Figure()
                            for j in range(len(bucket_labels)):
                                fig_opt_age.add_trace(go.Scatter(
                                    x=df["Date"], y=buckets_opt[:, j], name=bucket_labels[j],
                                    mode='lines', stackgroup='one', line=dict(width=0.5, color=bucket_colors[j])
                                ))
                            fig_opt_age.update_layout(template="plotly_white", yaxis_title="Units In Stock", xaxis_title="Date", height=350)
                            st.plotly_chart(fig_opt_age, use_container_width=True)

                    with drilldown_container:
                        st.markdown("---")
                        st.markdown("### 🔍 Point-in-Time Inventory Age Drilldown")
                        drilldown_date = st.date_input("Select specific date to inspect inventory age distribution", value=end_date, min_value=start_date, max_value=end_date)
                        
                        matched_row = df[df["Date"].dt.date == drilldown_date]
                        if not matched_row.empty:
                            idx_drill = matched_row.index[0]
                            act_dist = buckets_act[idx_drill]
                            opt_dist = buckets_opt[idx_drill]
                            
                            drill_df = pd.DataFrame({
                                "Age Bracket": bucket_labels,
                                "Historical Actuals (Units)": act_dist.astype(int),
                                "Optimized Policy (Units)": opt_dist.astype(int)
                            })
                            
                            col_chart, col_table = st.columns([2, 1])
                            with col_chart:
                                drill_fig = go.Figure()
                                drill_fig.add_trace(go.Bar(x=bucket_labels, y=act_dist, name="Actuals", marker_color="#B0C4DE"))
                                drill_fig.add_trace(go.Bar(x=bucket_labels, y=opt_dist, name="Optimized", marker_color="#1F77B4"))
                                drill_fig.update_layout(barmode='group', title=f"Age Distribution on {drilldown_date}", yaxis_title="Units", template="plotly_white", margin=dict(t=40, b=20))
                                st.plotly_chart(drill_fig, use_container_width=True)
                                
                            with col_table:
                                st.markdown(f"**Exact Stock Counts:**")
                                st.dataframe(drill_df, hide_index=True, use_container_width=True)

                    # ==========================================
                    #     SECTION B: COMPARATIVE ANALYSIS
                    # ==========================================
                    st.markdown("---")
                    st.header("🔬 Section B: Multi-Scenario Comparative Analysis")
                    st.markdown(
                        "Leveraging the high-speed vectorized simulation engine, you can now backtest and compare up to 6 "
                        "different inventory policies simultaneously. By adjusting these mechanical levers, you can easily identify "
                        "operational blind spots without manually tracking the math."
                    )
                    
                    active_scenarios_list = []
                    
                    tab_cont, tab_per = st.tabs(["📉 Continuous Review (Q, R) Scenarios", "⏳ Periodic Review (P, T) Scenarios"])
                    
                    with tab_cont:
                        st.markdown(f"**Baseline Intelligence:** The data-driven optimal benchmark is **Q: {int(raw_optimal_q):,}** and **ROP: {int(raw_target_level):,}**.")
                        
                        c1, c2, c3 = st.columns(3)
                        
                        def create_cont_box(col, num, default_q, default_r):
                            with col:
                                st.markdown(f"##### 🎛️ Scenario C{num}")
                                run_c = st.toggle(f"Include C{num} in Chart", key=f"run_c{num}", value=(num==1))
                                q_val = st.number_input("Order Qty (Q)", min_value=1, value=int(default_q), step=10, key=f"q_c{num}")
                                r_val = st.number_input("Reorder Point (ROP)", min_value=0, value=int(default_r), step=10, key=f"r_c{num}")
                                
                                if run_c:
                                    active_scenarios_list.append({
                                        "Case Name": f"C{num} (Q:{q_val}, R:{r_val})", 
                                        "Policy Type": "Continuous Review (Q, R)", 
                                        "P1": q_val, 
                                        "P2": r_val
                                    })

                        create_cont_box(c1, 1, raw_optimal_q, raw_target_level)
                        create_cont_box(c2, 2, raw_optimal_q * 1.5, raw_target_level)
                        create_cont_box(c3, 3, raw_optimal_q, raw_target_level * 1.2)

                    with tab_per:
                        st.markdown("**Baseline Intelligence:** Adjust the Review Period (P) below. The engine will dynamically calculate a safe expected Target Level (T) for that exact timeframe.")
                        
                        p1, p2, p3 = st.columns(3)
                        
                        def create_per_box(col, num, default_p):
                            with col:
                                st.markdown(f"##### 🎛️ Scenario P{num}")
                                run_p = st.toggle(f"Include P{num} in Chart", key=f"run_p{num}", value=False)
                                p_val = st.number_input("Review Period (P Days)", min_value=1, value=int(default_p), step=1, key=f"p_p{num}")
                                
                                target_guide = int(raw_target_level + (avg_daily_demand_calc * p_val))
                                
                                t_val = st.number_input("Target Level (T)", min_value=0, value=target_guide, step=10, key=f"t_p{num}")
                                st.caption(f"💡 *Engine recommended (T) for {p_val} days: **~{target_guide:,}***")
                                
                                if run_p:
                                    active_scenarios_list.append({
                                        "Case Name": f"P{num} (P:{p_val}, T:{t_val})", 
                                        "Policy Type": "Periodic Review (P, T)", 
                                        "P1": p_val, 
                                        "P2": t_val
                                    })

                        create_per_box(p1, 1, 7)
                        create_per_box(p2, 2, 14)
                        create_per_box(p3, 3, 30)

                    st.markdown("---")
                    
                    if st.button("🚀 Compare Scenarios", type="primary", use_container_width=True):
                        if not active_scenarios_list:
                            st.warning("Please toggle 'Include' for at least one scenario above to generate the comparison.")
                        else:
                            summary_data = []
                            comp_fig = go.Figure()
                            
                            comp_fig.add_trace(go.Scatter(
                                x=df["Date"], y=inv_levels_act, mode='lines', 
                                name="Historical Actuals", line=dict(color='rgba(176, 196, 222, 0.4)', width=2, dash='dot')
                            ))

                            summary_data.append({
                                "Scenario Blueprint": "📊 Historical Actuals (Baseline)",
                                "Total Op Cost ($)": actual_total_cost,
                                "Fill Rate (%)": actual_fill_rate * 100,
                                "Avg Inv (Units)": actual_avg_inventory,
                                "Avg Working Capital ($)": act_avg_wc,
                                "Max Peak Capital ($)": act_max_wc,
                                "Avg Age (Days)": np.mean(avg_age_act),
                                "Peak Age (Days)": np.max(max_age_act)
                            })

                            line_colors = px.colors.qualitative.D3
                            
                            for index, scenario in enumerate(active_scenarios_list):
                                case_name = scenario["Case Name"]
                                p_type = scenario["Policy Type"]
                                val1 = scenario["P1"]
                                val2 = scenario["P2"]
                                
                                s_inv, s_lost, s_orders, s_avg_age, s_max_age, s_buckets = fast_simulate_inventory(
                                    demand_arr_main, purchase_arr_main, opening_stock_override, 
                                    lead_time_days, p_type, val1, val2, custom_edges
                                )
                                
                                s_lost_sum = s_lost.sum()
                                s_orders_count = np.count_nonzero(s_orders)
                                s_avg_inv = np.mean(s_inv)
                                s_max_inv = np.max(s_inv)
                                
                                s_fill_rate = max(0.0, 1.0 - (s_lost_sum / max(1, total_demand)))
                                s_total_cost = (s_orders_count * ordering_cost) + (s_avg_inv * unit_holding_cost) + (s_lost_sum * lost_sales_penalty)
                                
                                summary_data.append({
                                    "Scenario Blueprint": case_name,
                                    "Total Op Cost ($)": s_total_cost,
                                    "Fill Rate (%)": s_fill_rate * 100,
                                    "Avg Inv (Units)": s_avg_inv,
                                    "Avg Working Capital ($)": s_avg_inv * item_unit_cost,
                                    "Max Peak Capital ($)": s_max_inv * item_unit_cost,
                                    "Avg Age (Days)": np.mean(s_avg_age),
                                    "Peak Age (Days)": np.max(s_max_age)
                                })
                                
                                comp_fig.add_trace(go.Scatter(
                                    x=df["Date"], y=s_inv, mode='lines', 
                                    name=case_name, line=dict(color=line_colors[index % len(line_colors)], width=2.5)
                                ))

                            st.markdown("##### 🏆 Comparative Outcomes Scorecard")
                            comp_df = pd.DataFrame(summary_data)
                            
                            def highlight_baseline(s):
                                return ['background-color: rgba(176, 196, 222, 0.15)' if s['Scenario Blueprint'] == "📊 Historical Actuals (Baseline)" else '' for v in s]

                            st.dataframe(
                                comp_df.style.apply(highlight_baseline, axis=1),
                                use_container_width=True, hide_index=True,
                                column_config={
                                    "Total Op Cost ($)": st.column_config.NumberColumn(format="$%.2f"),
                                    "Fill Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                                    "Avg Inv (Units)": st.column_config.NumberColumn(format="%d"),
                                    "Avg Working Capital ($)": st.column_config.NumberColumn(format="$%.0f"),
                                    "Max Peak Capital ($)": st.column_config.NumberColumn(format="$%.0f"),
                                    "Avg Age (Days)": st.column_config.NumberColumn(format="%.1f"),
                                    "Peak Age (Days)": st.column_config.NumberColumn(format="%.0f")
                                }
                            )
                            
                            st.markdown("##### 📈 Strategic Trajectory Matrix")
                            comp_fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                xaxis_title="Timeline Date", yaxis_title="On-Hand Inventory (Units)", 
                                height=450, legend=dict(orientation="h", y=1.1, x=1, xanchor="right")
                            )
                            st.plotly_chart(comp_fig, use_container_width=True)



# ==========================================
# NEW TAB: KPI & AGING ANALYSIS
# ==========================================
with tab7:
    st.header("Inventory KPI & Aging Analysis")
    
    # --- 1. Inputs ---
    st.subheader("1. Configuration & Data Upload")
    
    unit_value = st.number_input("Value Per Unit ($)", min_value=0.01, value=100.0, step=1.0, key="kpi_unit_val")
        
    uploaded_ledger = st.file_uploader("Upload Ledger (.xlsx or .csv) containing 'Date', 'Opening Balance', 'Demand/Sales', 'Receiving'", type=["xlsx", "csv"], key="kpi_uploader")
    
    if uploaded_ledger is not None:
        try:
            if uploaded_ledger.name.endswith('.csv'):
                df_kpi = pd.read_csv(uploaded_ledger)
            else:
                df_kpi = pd.read_excel(uploaded_ledger)
                
            df_kpi.columns = df_kpi.columns.str.strip()
            
            required_cols = ['Date', 'Opening Balance', 'Demand/Sales', 'Receiving']
            if not all(col in df_kpi.columns for col in required_cols):
                st.error(f"❌ The uploaded file must contain exactly these columns: {', '.join(required_cols)}")
            else:
                # Setup visual containers to control exactly where UI elements render
                kpi_container = st.container()
                chart_inv_container = st.container()
                chart_dem_container = st.container()
                bucket_input_container = st.container()
                chart_age_container = st.container()
                drilldown_container = st.container()
                table_container = st.container()

                # --- Data Processing & Daily Resampling ---
                df_kpi['Date'] = pd.to_datetime(df_kpi['Date'])
                df_kpi = df_kpi.sort_values(by="Date").reset_index(drop=True)
                
                # Extract starting balance dynamically from the first row of the file
                opening_stock = float(df_kpi['Opening Balance'].iloc[0])
                
                # Group by date to handle multiple entries on the same day
                df_kpi = df_kpi.groupby('Date').agg({'Demand/Sales': 'sum', 'Receiving': 'sum'}).reset_index()
                
                # Resample to daily frequency to ensure time-based FIFO aging increments correctly
                df_kpi = df_kpi.set_index('Date').resample('1D').asfreq().fillna(0).reset_index()
                
                # Place bucket input visually right before the age graph
                with bucket_input_container:
                    st.divider()
                    st.subheader("Inventory Aging Breakdown")
                    bucket_input = st.text_input("Define Age Buckets (Days, comma-separated)", value="30, 60, 90", key="kpi_buckets")
                
                # Parse Buckets
                try:
                    edges = sorted(list(set([int(x.strip()) for x in bucket_input.split(',') if x.strip().isdigit()])))
                    if not edges: edges = [30, 60, 90]
                except:
                    edges = [30, 60, 90]
                    
                labels = []
                prev = 0
                for e in edges:
                    labels.append(f"{prev}-{e} Days")
                    prev = e + 1
                labels.append(f"{prev}+ Days")
                
                num_buckets = len(labels)
                
                # --- FIFO Simulation Logic ---
                total_days = len(df_kpi)
                inv_levels = np.zeros(total_days)
                avg_ages = np.zeros(total_days)
                age_buckets_arr = np.zeros((total_days, num_buckets))
                
                current_inv = opening_stock
                fifo_queue = [[0, opening_stock]] if opening_stock > 0 else []
                
                for i in range(total_days):
                    dem = df_kpi['Demand/Sales'].iloc[i]
                    rec = df_kpi['Receiving'].iloc[i]
                    
                    if rec > 0:
                        fifo_queue.append([i, rec])
                        current_inv += rec
                        
                    demand_left = dem
                    while demand_left > 0 and fifo_queue:
                        if fifo_queue[0][1] <= demand_left:
                            demand_left -= fifo_queue[0][1]
                            fifo_queue.pop(0)
                        else:
                            fifo_queue[0][1] -= demand_left
                            demand_left = 0
                            
                    current_inv -= dem
                    if current_inv < 0: current_inv = 0
                    inv_levels[i] = current_inv
                    
                    if fifo_queue:
                        tot_q = 0
                        sum_age = 0
                        for item in fifo_queue:
                            arr_day, qty = item
                            age = i - arr_day
                            tot_q += qty
                            sum_age += (age * qty)
                            
                            idx = 0
                            while idx < len(edges) and age > edges[idx]:
                                idx += 1
                            age_buckets_arr[i, idx] += qty
                            
                        avg_ages[i] = sum_age / tot_q if tot_q > 0 else 0
                        
                df_kpi['Closing Balance'] = inv_levels
                df_kpi['Average Age (Days)'] = avg_ages
                for j, label in enumerate(labels):
                    df_kpi[label] = age_buckets_arr[:, j]
                    
                # --- 2. KPIs ---
                with kpi_container:
                    st.divider()
                    st.subheader("2. Inventory Performance Dashboard")
                    
                    min_inv = df_kpi['Closing Balance'].min()
                    max_inv = df_kpi['Closing Balance'].max()
                    avg_inv = df_kpi['Closing Balance'].mean()
                    
                    # Row 1: Physical Units
                    st.markdown("#### 📦 Physical Quantity")
                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric("Minimum Inventory", f"{int(min_inv):,} Units")
                    kpi2.metric("Maximum Inventory", f"{int(max_inv):,} Units")
                    kpi3.metric("Average Inventory", f"{int(avg_inv):,} Units")
                    
                    st.write("<br>", unsafe_allow_html=True) # Adds a little spacing
                    
                    # Row 2: Monetary Value
                    st.markdown("#### 💰 Financial Value")
                    val1, val2, val3 = st.columns(3)
                    val1.metric("Minimum Capital Tied Up", f"${min_inv * unit_value:,.0f}")
                    val2.metric("Peak Capital Tied Up", f"${max_inv * unit_value:,.0f}")
                    val3.metric("Average Capital Tied Up", f"${avg_inv * unit_value:,.0f}")


                
                # --- 3. Visual Diagnostics ---
                with chart_inv_container:
                    st.divider()
                    st.subheader("3. Visual Diagnostics")
                    st.markdown("#### Inventory Level Timeline")
                    fig_inv = go.Figure()
                    fig_inv.add_trace(go.Scatter(x=df_kpi['Date'], y=df_kpi['Closing Balance'], mode='lines', line=dict(color='#0673DF', width=2), name="Inventory Level", fill='tozeroy', fillcolor='rgba(6, 115, 223, 0.1)'))
                    fig_inv.update_layout(template="plotly_white", yaxis_title="Units", xaxis_title="Date", height=400, margin=dict(t=10, b=10))
                    st.plotly_chart(fig_inv, use_container_width=True)
                    
                with chart_dem_container:
                    st.markdown("#### Demand Distribution")
                    fig_hist = px.histogram(df_kpi[df_kpi['Demand/Sales'] > 0], x="Demand/Sales", nbins=20, color_discrete_sequence=['#0673DF'])
                    fig_hist.update_layout(template="plotly_white", yaxis_title="Frequency", xaxis_title="Daily Demand (Units)", height=400, margin=dict(t=10, b=10))
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                with chart_age_container:
                    st.markdown("#### FIFO Aging Profile")
                    fig_age = go.Figure()
                    
                    # Dynamically generate a blue-to-red scale based on the number of buckets
                    if num_buckets == 1:
                        colors = ['#0673DF']
                    else:
                        colors = px.colors.sample_colorscale('RdBu_r', [i/(num_buckets-1) for i in range(num_buckets)])
                    
                    for j, label in enumerate(labels):
                        fig_age.add_trace(go.Scatter(
                            x=df_kpi['Date'], y=df_kpi[label], name=label,
                            mode='lines', stackgroup='one', line=dict(width=1, color=colors[j])
                        ))
                    fig_age.update_layout(template="plotly_white", yaxis_title="Units In Stock", xaxis_title="Date", height=450, margin=dict(t=10, b=10))
                    st.plotly_chart(fig_age, use_container_width=True)
                    
                    st.markdown("#### Average Inventory Age")
                    fig_avg_age = go.Figure()
                    fig_avg_age.add_trace(go.Scatter(x=df_kpi['Date'], y=df_kpi['Average Age (Days)'], mode='lines', line=dict(color='#0673DF', width=2), name="Avg Age (Days)"))
                    fig_avg_age.update_layout(template="plotly_white", yaxis_title="Age (Days)", xaxis_title="Date", height=400, margin=dict(t=10, b=10))
                    st.plotly_chart(fig_avg_age, use_container_width=True)

                # --- 4. Point-in-Time Drilldown ---
                with drilldown_container:
                    st.divider()
                    st.markdown("### 🔍 Point-in-Time Inventory Age Drilldown")
                    
                    min_date = df_kpi['Date'].min().date()
                    max_date = df_kpi['Date'].max().date()
                    
                    drilldown_date = st.date_input("Select specific date to inspect inventory age distribution", value=max_date, min_value=min_date, max_value=max_date, key="kpi_drilldown_date")
                    
                    matched_row = df_kpi[df_kpi["Date"].dt.date == drilldown_date]
                    if not matched_row.empty:
                        # Extract the exact values for the selected day based on the labels
                        act_dist = [matched_row[label].iloc[0] for label in labels]
                        
                        drill_df = pd.DataFrame({
                            "Age Bracket": labels,
                            "Quantity (Units)": [int(x) for x in act_dist],
                            "Value ($)": [f"${x * unit_value:,.2f}" for x in act_dist]
                        })
                        
                        col_chart, col_table = st.columns([2, 1])
                        with col_chart:
                            drill_fig = go.Figure()
                            drill_fig.add_trace(go.Bar(x=labels, y=act_dist, marker_color="#0673DF"))
                            drill_fig.update_layout(title=f"Age Distribution on {drilldown_date}", yaxis_title="Units", template="plotly_white", margin=dict(t=40, b=20))
                            st.plotly_chart(drill_fig, use_container_width=True)
                            
                        with col_table:
                            st.markdown(f"**Exact Stock Counts:**")
                            st.dataframe(drill_df, hide_index=True, use_container_width=True)

                # --- 5. Tables ---
                with table_container:
                    st.divider()
                    st.subheader("4. Detailed Aging Table (End of Period Snapshot)")
                    
                    # Snapshot of the very last chronological day available
                    latest_date = df_kpi['Date'].iloc[-1]
                    latest_row = df_kpi.iloc[-1]
                    
                    age_data = []
                    total_units = latest_row['Closing Balance']
                    
                    for label in labels:
                        qty = latest_row[label]
                        age_data.append({
                            "Age Bracket": label,
                            "Quantity (Units)": int(qty),
                            "Value ($)": f"${qty * unit_value:,.2f}",
                            "% of Total Inventory": f"{(qty / total_units * 100) if total_units > 0 else 0:.1f}%"
                        })
                        
                    st.markdown(f"**Inventory Snapshot as of {latest_date.strftime('%Y-%m-%d')}**")
                    st.dataframe(pd.DataFrame(age_data), use_container_width=True, hide_index=True)
                    
                    with st.expander("📋 View Complete Daily Inventory Ledger"):
                        # Format output columns to clean up float displays before showing
                        display_df = df_kpi.copy()
                        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                        numeric_cols = display_df.select_dtypes(include=['float64']).columns
                        display_df[numeric_cols] = display_df[numeric_cols].round(1)
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
