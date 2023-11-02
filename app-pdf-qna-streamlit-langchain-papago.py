# E:\gdstrm\내 드라이브\Colab Notebooks\dl\dl_nlp\langchain\7_custom-llm_랭체인★★★\5_랭체인_PDF_업로드_Faiss_streamlit_유튜브★.ipynb

# 파파고는 하루 1만자 넘어가면 too many request라면서 에러가남



#  # 환경 설정 ===================================================

path_work = "."

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# 인증키
from dotenv import load_dotenv
load_dotenv()
import os
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
msy_naver_papago_api_id = os.getenv("msy_naver_papago_api_id")
msy_naver_papago_api_secret = os.getenv("msy_naver_papago_api_secret")


# 파파고 번역 함수 =============================================
def fn_papago(text, srcLang='en', tarLang='ko'):

    import os, sys, json
    import urllib.request

    encText = urllib.parse.quote(text)
    data = f"source={srcLang}&target={tarLang}&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",msy_naver_papago_api_id)
    request.add_header("X-Naver-Client-Secret",msy_naver_papago_api_secret)

    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        # print(response_body.decode('utf-8'))
        response_body_dict = json.loads(response_body.decode('utf-8'))
        response = response_body_dict['message']['result']['translatedText']
        # print(response)
    else:
        print("Error Code:" + rescode)
    
    return response


# 스트림잇 선언 =====================================================
import streamlit as st
st.set_page_config(page_title="Ask your PDF")
st.header("PDF QA ❓")
st.text("") 


# 임베딩 객체 생성 =====================================================
# [선택1]
embeddings = OpenAIEmbeddings()

# [선택2]
# from langchain.embeddings import HuggingFaceInstructEmbeddings
# embeddings = HuggingFaceInstructEmbeddings(
#     model_name="jhgan/ko-sroberta-multitask",
#     # model_name="sentence-transformers/all-MiniLM-L6-v2",
#     # model_kwargs={"device": "cpu"}
#     model_kwargs={"device": "cuda"}
#     )

# create embeddings
# embeddings = OpenAIEmbeddings()


# 생성 모델 로딩 ====================================================
# [선택1] 랭체인 모델 객체 생성 (openai) -------------------------------------
# from langchain.chat_models import ChatOpenAI
# # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# 
# [선택2] Custom LLM =========================================================
# 커스텀 LLM 클래스 함수
from pydantic import BaseModel, Field
from typing import Any, Optional, Dict, List
from huggingface_hub import InferenceClient
from langchain.llms.base import LLM

class KwArgsModel(BaseModel):
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class CustomInferenceClient(LLM, KwArgsModel):
    model_name: str
    inference_client: InferenceClient

    def __init__(self, model_name: str, hf_token: str, kwargs: Optional[Dict[str, Any]] = None):
        inference_client = InferenceClient(model=model_name, token=hf_token)
        super().__init__(
            model_name=model_name,
            hf_token=hf_token,
            kwargs=kwargs,
            inference_client=inference_client  # inference_client 인자 추가
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        response_gen = self.inference_client.text_generation(prompt, **self.kwargs, stream=True)
        response = ''.join(response_gen)  # 제너레이터의 모든 값을 문자열로 연결
        # response = self.inference_client.text_generation(prompt, **self.kwargs, stream=False)
        return response

    @property
    def _llm_type(self) -> str:
        return "custom"

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name}

# LLM 객체 생성
# 사용자로부터 모델 이름을 드롭다운 메뉴에서 선택받는 코드 추가
model_name = st.selectbox(  # 변경된 부분
    "LLM 변경 (GPU 한도로 인한 오류 발생시에만 변경, ※ Llama-2-70b가 가장 답변을 잘해요~)",
    (
        "meta-llama/Llama-2-70b-chat-hf",
        "tiiuae/falcon-180B-chat",
        "meta-llama/Llama-2-13b-chat-hf",
        "HuggingFaceH4/zephyr-7b-alpha"
    ),
    index=0  # 디폴트로 첫 번째 옵션 선택
)

kwargs = {"max_new_tokens":1024, "temperature":0.01, "top_p":0.6, "repetition_penalty":1.3, "do_sample":True}
llm = CustomInferenceClient(model_name=model_name, hf_token=hf_token, kwargs=kwargs)


# PDF 기반 QA ===================================================================================

# PDF 업로드
st.text("") 
pdf = st.file_uploader("PDF 파일을 업로드 하세요! (저장되지 않으니 걱정마세요~)", type="pdf")
st.markdown("[PDF 샘플 파일 다운로드 (상가임대차계약서)](https://www.moj.go.kr/sites/moj/download/316_01.pdf)")
st.text("")

# 질문 및 답변
if pdf is not None:

    # PDF 읽기
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # 청크로 분할
    text_splitter = CharacterTextSplitter(
    separator="\n",
    # chunk_size=800,
    chunk_size=400,
    chunk_overlap=200,
    length_function=len
    )
    chunks = text_splitter.split_text(text)

    # 벡터 DB에 저장
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # 스트림잇 질문 입력창과 버튼
    st.markdown('<style>.stButton>button { float: right; }</style>', unsafe_allow_html=True) # 버튼 오른쪽에 배치
    with st.form("question_form"):
        user_question = st.text_input(
            "PDF 내용 관련해서 궁금한 것을 질문하세요!",
            value="계약 갱신은 몇년 더 가능하지?"
        )
        submit_button = st.form_submit_button("Submit")

    # AI 및 번역 실행
    if user_question and submit_button:
        docs = knowledge_base.similarity_search(user_question, k=3, fetch_k=5)  # 5개 불러와서 3개 선택
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            try:
                # 생성 중 스피닝
                with st.spinner("AI가 확인중이예요! 조금만 기다려 주세요 ~"):
                    print("docs==>", docs)
                    response_original = chain.run(input_documents=docs, question=user_question)
                    print("response_original==>", response_original)
                response_original_cleaned = response_original.replace('</s>', '').replace('<|endoftext|>', '')
                print("response_original_cleaned ==>", response_original_cleaned)
                print(cb)

                # 파파고 번역
                try:
                    response_translated = fn_papago(response_original_cleaned, srcLang='en', tarLang='ko')
                    st.write(response_translated)
                    print("response_translated ==>", response_translated)
                except Exception as e: # 한도초과로 오류 발생시
                    st.write(response_original_cleaned + "\n\n (※ 번역 API 일 한도 초과되어 번역이 매끄럽지 못한 점 양해부탁드립니다.)")

            except Exception as e:
                st.error("해당 LLM의 일시적 GPU 사용 한도 초과로 다른 LLM으로 변경 선택하여 주세요!")
                print(str(e))  # 터미널에 예외 메시지를 출력합니다.
  
