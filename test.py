from pathlib import Path

print("CURRENT WORKING DIRECTORY:")
print(Path.cwd())

print("\nSCRIPT LOCATION (if available):")
try:
    print(Path(__file__).resolve())
except NameError:
    print("__file__ not available (you are likely in a notebook)")