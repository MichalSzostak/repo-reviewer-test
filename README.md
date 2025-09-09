infinityworks/data-academy-cloudformation-example — CFN template + lambda_function.py
https://github.com/infinityworks/data-academy-cloudformation-example

base2Services/cloudformation-custom-resources-python — CFN custom resources with Python handlers
https://github.com/base2Services/cloudformation-custom-resources-python

aws-samples/aws-lambda-deploy — CFN canary deploy patterns + Python Lambda code
https://github.com/aws-samples/aws-lambda-deploy

aws-cloudformation/aws-cloudformation-macros — CloudFormation macros implemented in Python
https://github.com/aws-cloudformation/aws-cloudformation-macros


Sample tickets (1 per repo)

(Assuming your four repos are:)

awslabs/aws-lambda-deploy (canary deploy tools)

base2Services/cloudformation-custom-resources-python (CFN custom resources in Python)

a CloudFormation templates repo (pure YAML/JSON stacks)

your Repo Explainer app (Module_4)

If these differ, tell me the names and I’ll swap the tickets to match.

1) awslabs/aws-lambda-deploy — safer rollout + observability
https://github.com/aws-samples/aws-lambda-deploy
Title: Harden canary rollout: structured logging, alarm-based rollback, & retention
Why: Improve safety/visibility of linear/canary rollouts.
Acceptance criteria:

Step Functions and Lambda functions emit structured JSON logs (request_id, alias, weights, version, decision).

CloudWatch Metric Filters and Alarms created for error rates & throttles; rollback path triggers on alarm.

CloudWatch log retention set (e.g., 30 or 90 days) via IaC.

No secrets/PII in logs; document fields that are allowed.

README adds a short “Observability & rollback” section (no CLI how-to).

2) base2Services/cloudformation-custom-resources-python — idempotent & least-privilege

Title: Make custom resource handlers idempotent with backoff & least-privilege IAM
Why: CFN invokes handlers multiple times; reduce drift & noisy failures.
Acceptance criteria:

CREATE/UPDATE/DELETE handlers are idempotent (dedupe by physical resource ID + stable key).

Implement exponential backoff + jitter on AWS SDK calls; retry only safe operations.

IAM policies for handlers tightened to least privilege (no wildcards).

Unit tests for idempotency, retries, and error mapping to CFN response.

Update docs: “Idempotency & retries” section with examples (no install steps).

3) CloudFormation templates repo — PCI-ish hardening pass

Title: Encrypt-by-default & tagging baseline across stacks
Why: Baseline that aligns with PCI-DSS friendly practices.
Acceptance criteria:

All S3, RDS/EBS, SNS/SQS resources have KMS encryption (CMK or AWS-managed as appropriate).

Log retention parameters for Lambda/LogGroups defined & used; VPC Flow Logs/Kinesis Firehose (if present) are encrypted.

Sensitive Parameters use NoEcho; outputs do not leak secrets/ARNs unnecessarily.

Required tags (e.g., Owner/Env/CostCenter/DataClass) attached to every resource via Mappings/StackSets/Modules.

Add cfn-lint and cfn-guard rules with a minimal CI check.

4) Repo Explainer (Module_4) — UX + performance pass

Title: Clean UX: show summaries in-app; sane defaults; fewer controls
Why: Streamline workflow and make results immediately visible.
Acceptance criteria:

UI shows summaries inline after “Summarise repo” (table or markdown).

Remove Refresh file list button; file list refreshes on clone/update.

Default CloudFormation model set (see UI change below).

Keep only sliders for model params (no number inputs); advanced params hidden or removed.

Add a one-line “truncation note” when input clipped.