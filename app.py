"""Local dashboard; Python standard-library HTTP server, no web framework needed."""
import argparse, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from model import available, experiment, predict
ROOT = Path(__file__).resolve().parent / 'web'
class Handler(BaseHTTPRequestHandler):
    def send_data(self, status, content, mime):
        self.send_response(status); self.send_header('Content-Type',mime); self.send_header('Content-Length',str(len(content))); self.end_headers(); self.wfile.write(content)
    def json(self, obj, status=200):
        self.send_data(status,json.dumps(obj,allow_nan=False).encode(),'application/json; charset=utf-8')
    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path=='/api/datasets': return self.json(available())
            if u.path=='/api/experiment':
                mode=parse_qs(u.query).get('mode',['synthetic'])[0]
                if mode not in available(): raise ValueError('Dataset unavailable. Run fetch_dataset.py first.')
                return self.json(experiment(mode)[0])
            files={'/':('index.html','text/html'),'/style.css':('style.css','text/css'),'/app.js':('app.js','application/javascript')}
            if u.path not in files: return self.json({'error':'Not found'},404)
            name,mime=files[u.path]; self.send_data(200,(ROOT/name).read_bytes(),mime+'; charset=utf-8')
        except (ValueError,KeyError,OSError) as e: self.json({'error':str(e)},400)
    def do_POST(self):
        if self.path!='/api/predict': return self.json({'error':'Not found'},404)
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=16384: raise ValueError('Invalid request length.')
            data=json.loads(self.rfile.read(n))
            if data['mode'] not in available(): raise ValueError('Dataset unavailable.')
            return self.json(predict(data['mode'],data['key'],data['values']))
        except (ValueError,KeyError,TypeError,OSError) as e: self.json({'error':str(e)},400)
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8501); args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'EduPredict is running at http://127.0.0.1:{args.port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
