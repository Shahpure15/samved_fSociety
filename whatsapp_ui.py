import streamlit as st
import pandas as pd
from whatsapp_mock import WhatsAppEnforcementLog, format_whatsapp_message

def render_whatsapp_panel(log: WhatsAppEnforcementLog):
    st.markdown('''
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <h3 style="margin: 0;">WhatsApp Enforcement Panel</h3>
            <div style="width: 14px; height: 14px; background-color: #25D366; border-radius: 50%; box-shadow: 0 0 5px #25D366;"></div>
        </div>
    ''', unsafe_allow_html=True)
    
    all_alerts = log.get_all()
    pending = log.get_pending()
    approved = [a for a in all_alerts if a.status == "approved"]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Alerts Sent", len(all_alerts))
    col2.metric("Approved", len(approved))
    col3.metric("Pending Review", len(pending))
    
    if len(all_alerts) == 0:
        st.info("No violations detected yet. Upload a video and violations will appear here automatically.")
        return
        
    st.markdown("#### Pending Actions")
    for alert in pending:
        msg = format_whatsapp_message(alert)
        msg_html = msg.replace('\n', '<br>')
        
        with st.container():
            st.markdown(
                f"""
                <div style="
                    background-color: #dcf8c6;
                    border-radius: 12px;
                    padding: 14px 18px;
                    margin-bottom: 10px;
                    font-family: monospace;
                    font-size: 0.9rem;
                    line-height: 1.5;
                    box-shadow: 1px 1px 4px rgba(0,0,0,0.15);
                ">
                {msg_html}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.caption("Sent to: Duty Roster — 4 officers")
            
            # Using unique keys for buttons
            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button("Approve E-Challan", key=f"app_{alert.vehicle_id}", type="primary", use_container_width=True):
                    log.approve(alert.vehicle_id)
                    st.success("E-Challan issued. Evidence package generated.")
                    st.rerun()
            with bc2:
                if st.button("Reject", key=f"rej_{alert.vehicle_id}", use_container_width=True):
                    log.reject(alert.vehicle_id)
                    st.warning("Violation rejected. No challan issued.")
                    st.rerun()
            
            st.markdown("---")
            
    with st.expander("Enforcement History"):
        history = [a for a in all_alerts if a.status != "pending"]
        if history:
            df = pd.DataFrame([{
                "Vehicle ID": a.vehicle_id,
                "Location": f"{a.location_name} ({a.camera_id})",
                "Duration": f"{a.duration_seconds // 60}m {a.duration_seconds % 60}s",
                "Status": a.status.upper(),
                "Action Taken": getattr(a, 'action_timestamp', a.timestamp)
            } for a in history])
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No history available.")