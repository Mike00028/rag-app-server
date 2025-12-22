celerey run command
tasks is filename without extension .py
```
celery -A tasks worker --loglevel=info --pool=threads
```

redis server in wsl
sudo systemctl start redis-server.service