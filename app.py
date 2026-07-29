import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
import streamlit_authenticator as stauth

# -----------------------------------------
# Page Configuration
# -----------------------------------------
st.set_page_config(page_title="Project Cash Flow & Profitability", layout="wide")

# -----------------------------------------
# Secure Authentication Module (v0.3.0+)
# -----------------------------------------
# Define user credentials directly (Make sure to replace passwords with your generated hashes)
config = {
    'credentials': {
        'usernames': {
            'admin': {
                'email': 'a@gamil.com',
                'name': 'System Admin',
                'password': '$2b$12$9cTEuIxNy2EF8sjqI/EFUObtti2opTHNwgdYie.seoAQAKYBRaFj2'  # e.g., '$2b$12$xyz...'
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
        'expiry_days': 30,
        'key': 'random_secret_signature_key_here', 
        'name': 'project_cashflow_cookie'
    }
}

# Initialize the authenticator (No 'preauthorized' parameter)
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Render the login widget
try:
    authenticator.login()
except Exception as e:
    st.error(e)

# Check the session state for authentication status
if st.session_state.get("authentication_status") is False:
    st.error("Username/password is incorrect")
    st.stop()
elif st.session_state.get("authentication_status") is None:
    st.warning("Please enter your username and password")
    st.stop()

# -----------------------------------------
# Main Application (Only visible if logged in)
# -----------------------------------------
st.title("Project Cash Flow & Profitability Assessment")

# Sidebar for Global Parameters & Logout
with st.sidebar:
    st.write(f"Welcome, **{st.session_state.get('name', 'User')}**")
    authenticator.logout("Log Out", "sidebar")
    
    st.header("Financial Parameters")
    total_project_value = st.number_input("Total Project Value", min_value=0.0, value=100000.0, step=1000.0)
    interest_rate_pct = st.number_input("Annual Interest Rate (%)", min_value=0.0, value=12.0, step=0.5)
    earn_interest = st.checkbox("Earn interest on positive balance?", value=False)

# -----------------------------------------
# Data Input Sections (Tabs)
# -----------------------------------------
tab1, tab2, tab3 = st.tabs(["Outflows (Costs)", "Inflows (Payments)", "Cash Flow & Profitability"])

# TAB 1: Outflows
with tab1:
    st.subheader("Component Fulfillment")
    
    if "comp_df" not in st.session_state:
        st.session_state.comp_df = pd.DataFrame(
            columns=["Component", "Purchase Date", "Lead Time (Days)", "Credit Period (Days)", "Cost", "Cash Outflow Date"]
        )
        st.session_state.comp_df.loc[0] = ["Raw Material A", datetime.date.today(), 14, 30, 15000.0, datetime.date.today() + datetime.timedelta(days=30)]

    components_df = st.data_editor(
        st.session_state.comp_df,
        num_rows="dynamic",
        column_config={
            "Purchase Date": st.column_config.DateColumn("Purchase Date", format="YYYY-MM-DD"),
            "Cash Outflow Date": st.column_config.DateColumn("Cash Outflow Date", format="YYYY-MM-DD"),
            "Cost": st.column_config.NumberColumn("Cost", format="$%f", min_value=0.0)
        },
        use_container_width=True,
        key="comp_editor"
    )
    
    st.markdown("---")
    st.subheader("Fixed Cost Contribution")
    
    if "fc_df" not in st.session_state:
        st.session_state.fc_df = pd.DataFrame(columns=["Description", "Fixed Cost", "Date"])
        st.session_state.fc_df.loc[0] = ["Overhead allocation", 5000.0, datetime.date.today()]

    fixed_costs_df = st.data_editor(
        st.session_state.fc_df,
        num_rows="dynamic",
        column_config={
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Fixed Cost": st.column_config.NumberColumn("Fixed Cost", format="$%f", min_value=0.0)
        },
        use_container_width=True,
        key="fc_editor"
    )

# TAB 2: Inflows
with tab2:
    st.subheader("Customer Payment Schedule")
    st.info(f"Target Total Project Value: **${total_project_value:,.2f}**")
    
    if "pay_df" not in st.session_state:
        st.session_state.pay_df = pd.DataFrame(columns=["Milestone/Day", "Date", "Value"])
        st.session_state.pay_df.loc[0] = ["Advance Payment", datetime.date.today(), 25000.0]

    payments_df = st.data_editor(
        st.session_state.pay_df,
        num_rows="dynamic",
        column_config={
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Value": st.column_config.NumberColumn("Absolute Value", format="$%f", min_value=0.0)
        },
        use_container_width=True,
        key="pay_editor"
    )
    
    if not payments_df.empty and total_project_value > 0:
        # Calculate percentage using total value
        payments_df['% of Total'] = (payments_df['Value'] / total_project_value) * 100
        st.dataframe(payments_df.style.format({'% of Total': '{:.2f}%', 'Value': '${:,.2f}'}), use_container_width=True)
        
        total_scheduled = payments_df['Value'].sum()
        variance = total_project_value - total_scheduled
        if variance != 0:
            st.warning(f"Note: Total scheduled payments (${total_scheduled:,.2f}) differ from Total Project Value by ${variance:,.2f}.")

# TAB 3: Cash Flow Logic & Profitability
with tab3:
    if st.button("Calculate Profitability & Generate Cash Flow", type="primary"):
        events = []
        
        # Consolidate Components
        for _, row in components_df.iterrows():
            if pd.notna(row['Cash Outflow Date']) and pd.notna(row['Cost']):
                events.append({'Date': pd.to_datetime(row['Cash Outflow Date']), 'Amount': -row['Cost'], 'Type': 'Component'})
                
        # Consolidate Fixed Costs
        for _, row in fixed_costs_df.iterrows():
            if pd.notna(row['Date']) and pd.notna(row['Fixed Cost']):
                events.append({'Date': pd.to_datetime(row['Date']), 'Amount': -row['Fixed Cost'], 'Type': 'Fixed Cost'})
                
        # Consolidate Payments (Inflows)
        for _, row in payments_df.iterrows():
            if pd.notna(row['Date']) and pd.notna(row['Value']):
                events.append({'Date': pd.to_datetime(row['Date']), 'Amount': row['Value'], 'Type': 'Payment Inflow'})
        
        if not events:
            st.warning("Please enter at least one financial event to calculate cash flow.")
        else:
            events_df = pd.DataFrame(events)
            events_df = events_df.sort_values('Date')
            
            # Generate Daily Timeline
            start_date = events_df['Date'].min()
            end_date = events_df['Date'].max()
            date_range = pd.date_range(start=start_date, end=end_date)
            
            timeline_df = pd.DataFrame({'Date': date_range})
            
            # Group events happening on the same day
            daily_events = events_df.groupby('Date')['Amount'].sum().reset_index()
            daily_events.rename(columns={'Amount': 'Daily Net Cash'}, inplace=True)
            
            timeline_df = pd.merge(timeline_df, daily_events, on='Date', how='left').fillna(0)
            
            # Interest logic
            daily_interest_rate = (interest_rate_pct / 100) / 365
            balances = []
            interest_charges = []
            running_balance = 0.0
            
            for index, row in timeline_df.iterrows():
                running_balance += row['Daily Net Cash']
                daily_interest = 0.0
                
                if running_balance < 0:
                    daily_interest = running_balance * daily_interest_rate 
                elif running_balance > 0 and earn_interest:
                    daily_interest = running_balance * daily_interest_rate
                
                balances.append(running_balance)
                interest_charges.append(daily_interest)
                
            timeline_df['Ending Balance'] = balances
            timeline_df['Daily Interest'] = interest_charges
            timeline_df['Cumulative Interest'] = timeline_df['Daily Interest'].cumsum()
            
            # KPI Calculations (Absolute values as preferred)
            total_inflows = events_df[events_df['Amount'] > 0]['Amount'].sum()
            total_outflows = abs(events_df[events_df['Amount'] < 0]['Amount'].sum())
            total_net_interest = timeline_df['Daily Interest'].sum()
            
            gross_profit = total_inflows - total_outflows
            net_profit = gross_profit + total_net_interest 
            
            # KPI Display
            st.subheader("Project Profitability Assessment")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Inflows", f"${total_inflows:,.2f}")
            col2.metric("Total Outflows", f"${total_outflows:,.2f}")
            col3.metric("Net Interest Paid/Earned", f"${total_net_interest:,.2f}")
            col4.metric("Net Project Profit", f"${net_profit:,.2f}")
            
            st.markdown("---")
            st.subheader("Cash Flow Timeline")
            
            # Blue-themed Plotly Chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timeline_df['Date'], 
                y=timeline_df['Ending Balance'],
                fill='tozeroy',
                mode='lines',
                line=dict(color='#1E88E5', width=2),
                name='Cash Balance',
                fillcolor='rgba(30, 136, 229, 0.2)'
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Balance")
            
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis_title="Date",
                yaxis_title="Amount ($)",
                hovermode="x unified",
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Daily Ledger
            with st.expander("View Daily Financial Ledger"):
                st.dataframe(timeline_df.style.format({
                    'Daily Net Cash': '${:,.2f}',
                    'Ending Balance': '${:,.2f}',
                    'Daily Interest': '${:,.2f}',
                    'Cumulative Interest': '${:,.2f}'
                }), use_container_width=True)
