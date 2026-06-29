@echo off
for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
  if /i "%%A"=="OPENAI_API_KEY" set "OPENAI_API_KEY=%%B"
  if /i "%%A"=="S3_BUCKET_NAME" set "S3_BUCKET_NAME=%%B"
)
start "" "https://console.cloud.google.com/apis/dashboard?project=youtube-api-484522&pageState=(%%22duration%%22:(%%22groupValue%%22:%%22P2D%%22,%%22customValue%%22:null))"
start "" "https://eu-west-3.console.aws.amazon.com/s3/buckets/rag-ionis-532523613357-eu-west-3-an?region=eu-west-3&tab=objects"
start "" "https://platform.openai.com/usage"
start "" "https://developers.openai.com/api/docs/pricing"
start "" "https://platform.openai.com/tokenizer"
