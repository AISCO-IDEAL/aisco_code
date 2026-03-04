from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os


def main() -> None:
	templates_dir = Path(__file__).resolve().parent / "templates"
	os.chdir(templates_dir)

	host = "0.0.0.0"
	port = 8000

	print(f"Serving GUI from {templates_dir}")
	print(f"WSL URL: http://127.0.0.1:{port}")
	print("Open in Windows Firefox:")
	print(f"  /mnt/c/Program\\ Files/Mozilla\\ Firefox/firefox.exe http://localhost:{port}")

	server = ThreadingHTTPServer((host, port), SimpleHTTPRequestHandler)
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		print("\nShutting down server.")
	finally:
		server.server_close()


if __name__ == "__main__":
	main()
