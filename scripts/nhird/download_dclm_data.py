import yaml
import requests
import os
from urllib.parse import urlparse

def read_urls_from_yaml(yaml_file_path):
    """Read URLs from the data.paths section of a YAML file."""
    try:
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)
            paths = data.get('data', {}).get('paths', [])
            urls = []
            for path in paths:
                if isinstance(path, str):
                    if path.startswith('http://') or path.startswith('https://'):
                        urls.append(path)
                elif isinstance(path, dict):
                    for value in path.values():
                        if isinstance(value, str) and (value.startswith('http://') or value.startswith('https://')):
                            urls.append(value)
            return urls
    except FileNotFoundError:
        print(f"Error: YAML file '{yaml_file_path}' not found.")
        return []
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def read_urls_from_txt(txt_file_path):
    """Read URLs from a text file where each line is a URL."""
    urls = []
    try:
        with open(txt_file_path, 'r') as file:
            for line in file:
                url = line.strip()
                if url and (url.startswith('http://') or url.startswith('https://')):
                    urls.append(url)
        return urls
    except FileNotFoundError:
        print(f"Error: Text file '{txt_file_path}' not found.")
        return []
    except Exception as e:
        print(f"Unexpected error reading text file: {e}")
        return []

def download_file(url, download_dir, failed_count):
    """Download a file from a URL to the specified directory, preserving the URL path structure."""
    try:
        parsed_url = urlparse(url)
        path = parsed_url.path.lstrip('/')
        if not path:
            path = 'downloaded_file'
        file_path = os.path.join(download_dir, parsed_url.netloc, path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            print(f"✅ Downloaded: {url} -> {file_path}")
        else:
            print(f"❌ Failed to download {url}: Status code {response.status_code}")
            failed_count[0] += 1
    except requests.RequestException as e:
        print(f"❌ Error downloading {url}: {e}")
        failed_count[0] += 1
    except Exception as e:
        print(f"❌ Unexpected error while downloading {url}: {e}")
        failed_count[0] += 1
    return failed_count

def main():
    # === CONFIGURATION ===
    yaml_file_path = './configs/1b_nhird.yaml'
    txt_file_path = './scripts/nhird/dclm_path_6B.txt'
    download_dir = '/tmp/olmo'
    
    use_yaml = False       # Set to True to use YAML
    use_txt = True        # Set to True to use TXT
    merge_urls = False     # If both sources are used, merge and remove duplicates

    failed_count = [0]

    urls = []

    # === LOAD URLS ===
    if use_yaml:
        yaml_urls = read_urls_from_yaml(yaml_file_path)
        print(f"\n🌐 Loaded {len(yaml_urls)} URLs from YAML")
        urls.extend(yaml_urls)

    if use_txt:
        txt_urls = read_urls_from_txt(txt_file_path)
        print(f"📄 Loaded {len(txt_urls)} URLs from TXT")
        if merge_urls:
            urls.extend(txt_urls)
        else:
            urls = txt_urls  # override if not merging

    # === REMOVE DUPLICATES ===
    urls = list(dict.fromkeys(urls))  # preserve order, remove duplicates

    if urls:
        print(f"\n📌 Total unique URLs to download: {len(urls)}")
        for url in urls:
            print(f"- {url}")
    else:
        print("\n⚠️ No valid URLs found. Exiting.")
        return

    # === DOWNLOAD FILES ===
    for url in urls:
        failed_count = download_file(url, download_dir, failed_count)

    print(f"\n🎉 Download complete. Number of failed files: {failed_count[0]}")

if __name__ == "__main__":
    main()
