# API Configuration

The frontend now uses a configurable API base URL instead of hardcoded IP addresses.

## Configuration Options

### Option 1: Environment Variable (Recommended)
Create or edit the `.env.local` file in the frontend directory:

```bash
# For remote server
NEXT_PUBLIC_API_BASE_URL=http://172.25.27.208:5000

# For local development
NEXT_PUBLIC_API_BASE_URL=http://localhost:5000
```

### Option 2: Default Fallback
If no environment variable is set, the frontend will default to `http://localhost:5000`.

## How to Change the API Server

1. **For Local Development:**
   ```bash
   echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:5000" > .env.local
   ```

2. **For Remote Server:**
   ```bash
   echo "NEXT_PUBLIC_API_BASE_URL=http://YOUR_SERVER_IP:5000" > .env.local
   ```

3. **Restart the Next.js development server** after changing the environment variable:
   ```bash
   npm run dev
   ```

## Current Configuration
The `.env.local` file is currently set to: `http://172.25.27.208:5000`

You can verify the current API base URL by checking the browser's network tab or console when making API requests.