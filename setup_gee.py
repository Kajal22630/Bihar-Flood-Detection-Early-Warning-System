"""
GEE OAuth Setup Helper for kumariannu8610@gmail.com
Run this script ONCE to authenticate, then restart app.py
"""
import subprocess
import sys

print("=" * 60)
print("  GOOGLE EARTH ENGINE — ONE-TIME AUTHENTICATION HELPER")
print("=" * 60)
print()
print("  Account : kumariannu8610@gmail.com")
print("  Step 1  : This will open your browser to authorize GEE")
print("  Step 2  : Sign in with kumariannu8610@gmail.com")
print("  Step 3  : Copy the auth code back here")
print()

try:
    import ee
    ee.Authenticate()
    # Test initialization
    try:
        ee.Initialize(project='ee-kumariannu8610')
        print("\n  ✅ GEE Authenticated with project: ee-kumariannu8610")
    except Exception as e1:
        ee.Initialize()
        print("\n  ✅ GEE Authenticated (default project)")
    print("  ✅ You can now restart app.py — SAR images will be REAL")
except ImportError:
    print("  ❌ earthengine-api not installed.")
    print("  Run:  pip install earthengine-api")
except Exception as e:
    print(f"  ❌ Authentication failed: {e}")
    print("  Try running:  earthengine authenticate")
    print("  Then restart app.py")
