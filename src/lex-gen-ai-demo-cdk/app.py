
import aws_cdk as cdk


from lex_gen_ai_demo_cdk_files.lex_gen_ai_demo_cdk_files_stack import LexGenAIDemoFilesStack
from create_web_crawler_lambda import LambdaStack
from endpoint_handler import  create_endpoint_from_JS_image

create_endpoint_from_JS_image(js_model_id="huggingface-llm-falcon-7b-instruct-bf16")

app = cdk.App()
file_stack = LexGenAIDemoFilesStack(app, "LexGenAIDemoFilesStack")
web_crawler_lambda_stack = LambdaStack(app, 'LexGenAIDemoFilesStack-Webcrawler')

app.synth()
