# Bare Metal Deployment Guide (Ubuntu/Debian)

This guide explains how to run the application directly on a Linux server without Docker, including Nginx for HTTPS.

## Prerequisites
- Python 3.11 or newer
- Git
- Access to the repository
- Domain DNS pointing to the server

## 1. System Dependencies
Install Python, Nginx, and Certbot:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv git build-essential nginx certbot python3-certbot-nginx
```

## 2. Clone Repository
```bash
git clone -b dev https://github.com/Silas-Asamoah/adom-tear-film-theoretical-spectra-generator.git
cd adom-tear-film-theoretical-spectra-generator
```

## 3. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-linux.txt
```

## 4. Run Application as Service
Create `/etc/systemd/system/pyelli.service`:
```ini
[Unit]
Description=PyElli Streamlit App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/adom-tear-film-theoretical-spectra-generator
ExecStart=/home/ubuntu/adom-tear-film-theoretical-spectra-generator/venv/bin/streamlit run exploration/pyelli_exploration/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless=true
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable pyelli
sudo systemctl start pyelli
```

## 5. Configure Nginx (Reverse Proxy)
Create `/etc/nginx/sites-available/pyelli`:
```nginx
server {
    listen 80;
    server_name dev.adom.reallygreattech.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/pyelli /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

## 6. Setup SSL (HTTPS)
Run Certbot to automatically configure SSL:
```bash
sudo certbot --nginx -d dev.adom.reallygreattech.com
```
Follow the prompts (enter email, agree to TOS).

## 7. Verify
Access https://dev.adom.reallygreattech.com
