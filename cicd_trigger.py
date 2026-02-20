
import boto3, json
sm = boto3.client("sagemaker")

response = sm.start_pipeline_execution(
    PipelineName="LiverDiseasePipeline",
    PipelineExecutionDisplayName="cicd-triggered",
    PipelineParameters=[
        {"Name": "ModelApprovalStatus",  "Value": "Approved"},
        {"Name": "AccuracyThreshold",    "Value": "0.75"},
    ]
)
execution_arn = response["PipelineExecutionArn"]
print(f"Pipeline execution started: {execution_arn}")

# Write ARN for next step
with open("execution_arn.txt", "w") as f:
    f.write(execution_arn)
