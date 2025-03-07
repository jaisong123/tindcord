# Deployment Guide for tindcord with Claude API

This guide provides instructions for deploying the tindcord Discord bot with Claude API integration on Google Cloud Platform (GCP).

## Prerequisites

1. A Google Cloud Platform account
2. Discord bot token and client ID
3. Anthropic Claude API key
4. Basic familiarity with Docker and GCP

## Option 1: Deployment with Cloud Run

Cloud Run is a fully managed platform that automatically scales stateless containers. It's ideal for the tindcord bot as it's lightweight and doesn't require persistent storage.

### Step 1: Set up Google Cloud SDK

1. Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Initialize the SDK and authenticate:
   ```bash
   gcloud init
   gcloud auth login
   ```

### Step 2: Create a config.yaml file

1. Copy the `config.yaml.sample` file to `config.yaml`
2. Fill in your Discord bot token, client ID, and Anthropic API key

### Step 3: Build and Push the Docker Image

1. Build the Docker image:
   ```bash
   docker build -t gcr.io/[YOUR_PROJECT_ID]/tindcord .
   ```

2. Push the image to Google Container Registry:
   ```bash
   gcloud auth configure-docker
   docker push gcr.io/[YOUR_PROJECT_ID]/tindcord
   ```

### Step 4: Deploy to Cloud Run

1. Deploy the container:
   ```bash
   gcloud run deploy tindcord \
     --image gcr.io/[YOUR_PROJECT_ID]/tindcord \
     --platform managed \
     --region [REGION] \
     --allow-unauthenticated \
     --memory 512Mi
   ```

2. Set up environment variables (optional, if you prefer not to include API keys in the config.yaml):
   ```bash
   gcloud run services update tindcord \
     --set-env-vars="ANTHROPIC_API_KEY=[YOUR_API_KEY],DISCORD_BOT_TOKEN=[YOUR_BOT_TOKEN]"
   ```

## Option 2: Deployment with Compute Engine

For more control over the environment, you can use a small Compute Engine VM.

### Step 1: Create a VM Instance

1. Go to the GCP Console > Compute Engine > VM Instances
2. Click "Create Instance"
3. Choose a small machine type (e.g., e2-micro)
4. Select your preferred region and zone
5. Choose a boot disk (e.g., Debian or Ubuntu)
6. Allow HTTP/HTTPS traffic if needed
7. Click "Create"

### Step 2: Set Up the VM

1. SSH into the VM:
   ```bash
   gcloud compute ssh [INSTANCE_NAME]
   ```

2. Install Docker:
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io
   sudo systemctl enable docker
   sudo systemctl start docker
   ```

3. Install Docker Compose:
   ```bash
   sudo apt-get install -y docker-compose
   ```

### Step 3: Deploy the Bot

1. Clone the repository:
   ```bash
   git clone https://github.com/jaisong123/tindcord.git
   cd tindcord
   ```

2. Create and configure the config.yaml file:
   ```bash
   cp config-example.yaml config.yaml
   nano config.yaml  # Edit with your credentials
   ```

3. Run with Docker Compose:
   ```bash
   sudo docker-compose up -d
   ```

## Monitoring and Maintenance

### Logging

1. View logs in Cloud Run:
   ```bash
   gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=tindcord"
   ```

2. View logs in Compute Engine:
   ```bash
   sudo docker logs -f tindcord_container_1
   ```

### Setting Up Budget Alerts

1. Go to Billing > Budgets & Alerts
2. Create a new budget with appropriate thresholds
3. Set up email notifications for budget alerts

### Updating the Bot

1. Pull the latest changes:
   ```bash
   git pull
   ```

2. Rebuild and restart the container:
   ```bash
   sudo docker-compose down
   sudo docker-compose up -d --build
   ```

## Troubleshooting

### Common Issues

1. **Bot not responding**: Check the logs for errors and ensure the Discord bot token is correct.
2. **Claude API errors**: Verify your API key and check for rate limiting issues.
3. **Container crashes**: Check for memory issues or configuration problems.

### Getting Help

If you encounter issues, check:
- The Discord.py documentation
- Anthropic's Claude API documentation
- The tindcord GitHub repository issues section

## Security Considerations

1. **API Keys**: Never commit API keys to the repository. Use environment variables or secure configuration.
2. **Firewall Rules**: Restrict access to your VM to only necessary ports.
3. **Regular Updates**: Keep the system and dependencies updated to patch security vulnerabilities.