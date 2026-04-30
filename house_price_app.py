import streamlit as st
import pickle 
import pandas as pd

#confg page
st.set_page_config(page_title="house price prediction",page_icon= "📚",layout="centered")

st.title("house price prediction app")

#load data
model = pickle.load(open("house_model.pkl","rb"))

#layout(side-by-side input and output)
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area in sqrft :",min_value=100,max_value=20000000,value=1000,step=50)
    bedrooms = st.number_input(" Number of bedrooms :",min_value=1,max_value=100,value=2)
    stories = st.number_input("Enter number of stories :",min_value=1,max_value=4,value=1)
    parking = st.number_input("parking space",min_value=0,max_value=5,value=1)


with col2:
    mainroad = st.selectbox("Main Road Access",["No","Yes"],index=1)
    guestroom = st.selectbox("Guest Room",["No","Yes"],index=0)
    basement = st.selectbox("Basement",["No","Yes"],index=0)



#convert yes/no to 1/0
mainroad_val = 1 if mainroad == "Yes" else 0
guestroom_val = 1 if guestroom == "Yes" else 0
basement_val = 1 if basement == "Yes" else 0




if st.button("Predict price",type="primary"):
    # order must match training data columns exactly
    input_data = pd.DataFrame([[area,bedrooms,stories,mainroad_val,guestroom_val,basement_val,parking]],columns=["area", "bedrooms", "stories", "mainroad",
        "guestroom", "basement", "parking"])
    pred = model.predict(input_data)[0]

    #show result
    st.success(f"Estimate House Price:₹ {pred:,.0f}")
    st.balloons()

    with st.expander("View input data sent to model"):
        st.dataframe(input_data)
    

#footer
st.markdown("---")
st.caption("Created by [Ahmad Afnan]")





























