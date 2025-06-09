from supabase import create_client, Client
import requests
import tempfile

SUPABASE_URL = "https://wprpxucfksxevfaxeskv.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndwcnB4dWNma3N4ZXZmYXhlc2t2Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0ODk2NzkxMCwiZXhwIjoyMDY0NTQzOTEwfQ.NMCYGUMXHLUW0fnW1fb3v1x4lqZEMpGOI0ucr6Gt1WE" 

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

async def getFileBinaryData(path:str, bucket:str):
  signed_url_response = supabase.storage.from_(bucket).create_signed_url(
    path=path,
    expires_in=3600  # seconds
  )

  download_url = signed_url_response.get("signedURL")
  response = requests.get(download_url)
  
  print(response)
  return response.content

async def uploadFile(filePath, storagePath):
  with open(filePath, "rb") as f:
    file_data = f.read()

  supabase.storage.from_("rezult").upload(path=storagePath, file=file_data)

def createTempFile(fileBytes, extension):
  with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
    temp_file.write(fileBytes)
    temp_file_path = temp_file.name
  
  return temp_file_path