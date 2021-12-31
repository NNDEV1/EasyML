import streamlit as st
from multiapp import MultiApp
from pages.automl import automl
from pages.idleeda import idle_eda
from pages.home import home
from pages.lazycompare import lazy_compare

st.set_page_config(page_title='EasyML',
                   layout='wide')

app = MultiApp()

# Add all your application here
app.add_app("Home", home)
app.add_app("IdleEDA", idle_eda)
app.add_app("AutoML", automl)
app.add_app("LazyCompare", lazy_compare)

# The main app
app.run()
