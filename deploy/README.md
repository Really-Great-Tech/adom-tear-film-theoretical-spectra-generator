# AWS Deployment Guide for PyElli Streamlit App

This guide covers setting up AWS infrastructure for deploying the PyElli app with GitHub Actions OIDC authentication.

## Prerequisites

- AWS CLI configured with appropriate permissions
- GitHub repository with Actions enabled
- Domain name pointed to your EC2 instance

## 1. Create ECR Repository

```bash
aws ecr create-repository \
    --repository-name pyelli-app \
    --region eu-central-1 \
    --profile delpho
```

## 2. Set Up GitHub OIDC Identity Provider

### Create the OIDC Provider

```bash
aws iam create-open-id-connect-provider \
    --url https://token.actions.githubusercontent.com \
    --client-id-list sts.amazonaws.com \
    --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 \
    --profile delpho
```

### Create IAM Role for GitHub Actions

Create a file `trust-policy.json`:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::YOUR_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
                },
                "StringLike": {
                    "token.actions.githubusercontent.com:sub": "repo:Silas-Asamoah/adom-tear-film-theoretical-spectra-generator:*"
                }
            }
        }
    ]
}
```

Create the role:

```bash
aws iam create-role \
    --role-name GitHubActions-PyElliDeploy \
    --assume-role-policy-document file://trust-policy.json \
    --profile delpho
```

### Attach ECR Permissions

```bash
aws iam attach-role-policy \
    --role-name GitHubActions-PyElliDeploy \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser \
    --profile delpho
```

## 3. EC2 Instance Setup

### Required Software (install manually)

```bash
# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu

# AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install

# Create app directory
sudo mkdir -p /opt/pyelli-app
sudo chown ubuntu:ubuntu /opt/pyelli-app
```

### Copy docker-compose.yml to EC2

```bash
scp docker-compose.yml ubuntu@YOUR_EC2_IP:/opt/pyelli-app/
```

### Create .env file on EC2

```bash
# /opt/pyelli-app/.env
DOMAIN=your-domain.com
ACME_EMAIL=your-email@example.com
```

### EC2 IAM Role

Attach an IAM role to the EC2 instance with ECR pull permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:GetAuthorizationToken"
            ],
            "Resource": "*"
        }
    ]
}
```

## 4. GitHub Secrets

Add these secrets in GitHub repository settings (Settings → Secrets → Actions):

| Secret | Value |
|--------|-------|
| `AWS_ROLE_ARN` | `arn:aws:iam::YOUR_ACCOUNT_ID:role/GitHubActions-PyElliDeploy` |
| `EC2_HOST` | Your EC2 public IP or domain |
| `EC2_SSH_KEY` | Contents of your SSH private key (PEM format) |

## 5. Security Group Rules

| Type | Port | Source |
|------|------|--------|
| SSH | 22 | Your IP |
| HTTP | 80 | 0.0.0.0/0 |
| HTTPS | 443 | 0.0.0.0/0 |

## 6. First Deployment

After pushing to the `dev` branch, the GitHub Action will:

1. Build the Docker image
2. Push to ECR
3. SSH to EC2 and pull the new image
4. Restart containers with Traefik

## 7. Manual Deployment (if needed)

```bash
# On EC2
cd /opt/pyelli-app

# Login to ECR
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.eu-central-1.amazonaws.com

# Pull and run
docker compose pull
docker compose up -d
```

## Troubleshooting

### Check container logs
```bash
docker logs pyelli-app
docker logs traefik
```

### Check SSL certificate status
```bash
docker exec traefik cat /letsencrypt/acme.json | jq '.letsencrypt.Certificates'
```

### Restart services
```bash
docker compose restart
```
