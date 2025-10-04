"""
Check and manage your OpenAI batch jobs
"""
from openai import OpenAI
import os

API_KEY = os.environ.get('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")
client = OpenAI(api_key=API_KEY)


# List all batches
print("Checking your batches...\n")
batches = client.batches.list(limit=10)

if not batches.data:
    print("No batches found.")
else:
    for batch in batches.data:
        print(f"Batch ID: {batch.id}")
        print(f"  Status: {batch.status}")
        print(f"  Created: {batch.created_at}")
        print(f"  Model: {batch.endpoint}")
        
        if batch.request_counts:
            print(f"  Progress: {batch.request_counts.completed}/{batch.request_counts.total}")
        
        # Show errors if failed
        if batch.status == "failed" and hasattr(batch, 'errors'):
            print(f"  ⚠️  Error: {batch.errors}")
        
        # Offer to cancel stuck or failed batches
        if batch.status in ["validating", "in_progress", "failed"]:
            action = input(f"  Cancel/clean up this batch? (y/n): ").lower()
            if action == 'y':
                try:
                    client.batches.cancel(batch.id)
                    print(f"  ✓ Cancelled {batch.id}")
                except Exception as e:
                    print(f"  ⚠️  Could not cancel: {e}")
        
        print()

print("\n" + "="*70)
print("Summary:")
print("="*70)
print(f"Total batches found: {len(batches.data)}")

status_counts = {}
for batch in batches.data:
    status_counts[batch.status] = status_counts.get(batch.status, 0) + 1

for status, count in status_counts.items():
    print(f"  {status}: {count}")