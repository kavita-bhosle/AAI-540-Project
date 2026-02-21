
import boto3, time

sm = boto3.client("sagemaker")
with open("execution_arn.txt") as f:
    arn = f.read().strip()

for _ in range(60):        # poll for up to 60 minutes
    r      = sm.describe_pipeline_execution(PipelineExecutionArn=arn)
    status = r["PipelineExecutionStatus"]
    print(f"Status: {status}")
    if status in ("Succeeded", "Failed", "Stopped"):
        break
    time.sleep(60)

if status != "Succeeded":
    raise SystemExit(f"Pipeline did not succeed: {status}")
print("Pipeline succeeded! Model ready for deployment.")
