import psutil

# Find and kill processes with the name "python eval_img_gen.py"
for process in psutil.process_iter(['name', 'cmdline']):
    if process.info['name'] == 'python' and 'app.py' in process.info['cmdline']:
        process.kill()