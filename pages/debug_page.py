
import streamlit as st
import config

st.title("Debug Session State")

if st.button("Reload Config"):
    import importlib
    importlib.reload(config)
    st.success(f"Config reloaded. LLM_MODEL_ID: {config.LLM_MODEL_ID}")

st.write("### Session State Keys")
st.write(list(st.session_state.keys()))

for key in st.session_state.keys():
    val = st.session_state[key]
    if hasattr(val, "model"):
        st.write(f"**Agent: {key}**")
        st.write(f"- Model Object: {type(val.model)}")
        if hasattr(val.model, "id"):
            st.write(f"- Model ID: `{val.model.id}`")
            if val.model.id == "llama-3.1-70b-versatile":
                st.error("FOUND LEGACY MODEL!")
        else:
            st.write("- Model ID: NOT FOUND")
    elif key == "model_id":
        st.write(f"**model_id**: `{val}`")

if st.button("CLEAR ALL AGENTS"):
    keys_to_del = [k for k in st.session_state.keys() if "agent" in k.lower() or k == "model_id" or k == "agents_initialized"]
    for k in keys_to_del:
        del st.session_state[k]
    st.rerun()
