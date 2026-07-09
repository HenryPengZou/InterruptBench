import boto3
import os

# os.environ['AWS_BEARER_TOKEN_BEDROCK'] = "<your_aws_bedrock_bearer_token>"

# Create the Bedrock client
client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-2" # us-east-1
)

# Define the model and message
# model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
# model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
# model_id = "us.meta.llama3-1-8b-instruct-v1:0"
# model_id = "meta.llama3-70b-instruct-v1:0"
# model_id = "qwen.qwen3-32b-v1:0"
# model_id = "deepseek.v3-v1:0"


messages = [{"role": "user", "content": [{"text": "Hello! Which model you are"}]}]

# Make the API call
response = client.converse(
    modelId=model_id,
    messages=messages,
)

# Print the response
print(response)
print(response['output']['message']['content'][0]['text'])
