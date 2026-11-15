/* Public configuration only. NEVER put a secret or API token in this file.
 * Same-origin (Flask /web/) default:
 *   window.LAWAI_API_BASE_URL = '';
 * Separate static frontend host (CORS mode):
 *   window.LAWAI_API_BASE_URL = 'https://your-backend.example.com';
 * The backend MUST explicitly allow the frontend origin in LAWAI_ALLOWED_ORIGINS.
 */
window.LAWAI_API_BASE_URL = '';
