import validators, streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain_core.prompts import PromptTemplate

# streamlit app
st.set_page_config(page_title="Langchain: Summarize text from YT", page_icon="🦜️")
st.title("🦜️ Summarize text from YT or website")
st.subheader("Summarize URL")

#get the groq api key and url
groq_api_key = st.sidebar.text_input("GROQ_API_KEY", type = "password", value="")
generic_url=st.text_input("URL",label_visibility="collapsed")

#basic prompt template
prompt_template="""Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


if st.button("summarize the content in this url"):
    #validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("please provide the information")
    elif not validators.url(generic_url):
        st.error("please provide valid url")
    else:
        try:
            with st.spinner("Waiting..."):
                #loading the website website or yt video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False)
                docs=loader.load()

                #chain for summarization
                llm = ChatGroq(groq_api_key = groq_api_key , model_name = "llama-3.1-8b-instant", streaming=True)
                chain = load_summarize_chain(llm,chain_type="stuff", prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")        
                






