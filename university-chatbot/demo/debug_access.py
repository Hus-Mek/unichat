"""
Debug Script - Check Access Control
Run this locally to verify metadata is stored correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import RAGEngine, AccessController

# Initialize
rag = RAGEngine()
access = AccessController()

# Get all documents
print("=" * 60)
print("ALL DOCUMENTS IN DATABASE")
print("=" * 60)

try:
    all_data = rag.collection.get()
    
    if not all_data["ids"]:
        print("❌ No documents in database!")
    else:
        for i, doc_id in enumerate(all_data["ids"]):
            metadata = all_data["metadatas"][i]
            print(f"\n📄 Document ID: {doc_id}")
            print(f"   Access Level: {metadata.get('access_level', 'MISSING!')}")
            print(f"   Owner: {metadata.get('owner', 'NONE')}")
            print(f"   Source: {metadata.get('source', 'UNKNOWN')}")
            print(f"   Chunk: {metadata.get('chunk_index', 'N/A')}")
        
        print(f"\n\nTotal chunks: {len(all_data['ids'])}")
        
        # Test access filters
        print("\n" + "=" * 60)
        print("ACCESS FILTER TESTS")
        print("=" * 60)
        
        for user_level in ["public", "student", "faculty"]:
            filter_dict = access.build_filter(user_level)
            print(f"\n{user_level.upper()} user filter:")
            print(f"  {filter_dict}")
            
            # Count how many documents this user can access
            result = rag.collection.query(
                query_embeddings=[rag.embedder.encode(["test"])[0].tolist()],
                n_results=100,
                where=filter_dict
            )
            accessible_count = len(result["documents"][0])
            print(f"  Can access: {accessible_count} chunks")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
