"""
Setup script to initialize the Mini RAG system
"""
import os
import sys
from pdf_processor import PDFProcessor
from vector_search import VectorSearch
from config import PDF_DIR, FAISS_INDEX_PATH, DATABASE_PATH


def create_sample_pdfs():
    """Create sample PDF content for testing (since we don't have the actual ZIP file)"""
    os.makedirs(PDF_DIR, exist_ok=True)
    
    # Sample content for testing
    sample_content = {
        "ISO_13849-1_2015.pdf": """
        ISO 13849-1:2015 - Safety of machinery - Safety-related parts of control systems
        
        This international standard specifies safety requirements for safety-related parts of control systems. 
        It provides guidance on the design and integration of safety-related parts of control systems.
        
        The standard defines performance levels (PL) from PLa to PLe, where PLe represents the highest level of safety integrity.
        Performance levels are determined based on the combination of safety integrity level (SIL) and mean time to dangerous failure (MTTFd).
        
        Key requirements include:
        - Systematic capability (SC) requirements
        - Hardware fault tolerance (HFT) requirements  
        - Diagnostic coverage (DC) requirements
        - Common cause failure (CCF) requirements
        
        The standard applies to safety-related parts of control systems that are used to perform safety functions.
        """,
        
        "IEC_61508_2010.pdf": """
        IEC 61508:2010 - Functional safety of electrical/electronic/programmable electronic safety-related systems
        
        This standard provides a generic approach for all safety lifecycle activities for systems that are used to perform safety functions.
        It covers the safety lifecycle from initial concept through decommissioning.
        
        The standard defines four safety integrity levels (SIL 1 to SIL 4), where SIL 4 represents the highest level of safety integrity.
        Each SIL has specific requirements for:
        - Hardware safety integrity
        - Systematic safety integrity
        - Software safety integrity
        
        Key concepts include:
        - Safety function: Function to be implemented by an E/E/PE safety-related system
        - Safety integrity: Probability of a safety-related system satisfactorily performing the required safety functions
        - Safety lifecycle: All activities necessary to achieve functional safety
        """,
        
        "OSHA_Machine_Guarding.pdf": """
        OSHA Machine Guarding Safety Requirements
        
        Machine guarding is essential for protecting workers from machine hazards. OSHA requires that machines be guarded to prevent worker injury.
        
        Types of machine guards include:
        - Fixed guards: Permanent barriers that prevent access to danger zones
        - Interlocked guards: Guards that automatically shut off the machine when opened
        - Adjustable guards: Guards that can be adjusted for different operations
        - Self-adjusting guards: Guards that move with the work piece
        
        General requirements:
        - Guards must prevent contact with moving parts
        - Guards must be secure and not create new hazards
        - Guards must allow for safe operation and maintenance
        - Emergency stop devices must be readily accessible
        
        Common machine hazards include:
        - Point of operation hazards
        - In-running nip point hazards
        - Rotating parts hazards
        - Flying chips and sparks
        """,
        
        "EN_ISO_12100_2010.pdf": """
        EN ISO 12100:2010 - Safety of machinery - General principles for design - Risk assessment and risk reduction
        
        This standard provides general principles for the design of machinery and guidance for risk assessment and risk reduction.
        It establishes a systematic approach to machinery safety.
        
        The risk assessment process includes:
        1. Determination of the limits of the machinery
        2. Hazard identification
        3. Risk estimation
        4. Risk evaluation
        5. Risk reduction
        
        Risk reduction follows the three-step method:
        1. Inherently safe design measures
        2. Safeguarding and/or complementary protective measures
        3. Information for use
        
        Key principles:
        - Safety by design is the most effective approach
        - Risk assessment should be iterative
        - Residual risks must be communicated to users
        - Safety measures should not create new hazards
        """,
        
        "NFPA_79_2018.pdf": """
        NFPA 79:2018 - Electrical Standard for Industrial Machinery
        
        This standard covers electrical/electronic equipment of industrial machines operating from a nominal voltage of 600 volts or less.
        It provides requirements for the electrical equipment of machines to promote safety and reduce fire hazards.
        
        Key requirements include:
        - Electrical equipment must be suitable for its intended use
        - Equipment must be properly installed and maintained
        - Electrical connections must be secure and protected
        - Grounding and bonding requirements must be met
        
        Safety requirements:
        - Emergency stop circuits must be fail-safe
        - Control circuits must be properly designed
        - Electrical enclosures must provide adequate protection
        - Cable management systems must be properly installed
        
        The standard also covers:
        - Motor control circuits
        - Lighting and receptacle circuits
        - Special equipment and systems
        - Maintenance and testing requirements
        """
    }
    
    # Create sample text files (simulating PDFs)
    for filename, content in sample_content.items():
        filepath = os.path.join(PDF_DIR, filename.replace('.pdf', '.txt'))
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"Created {len(sample_content)} sample documents in {PDF_DIR}")


def setup_system():
    """Complete system setup"""
    print("Setting up Mini RAG + Reranker system...")
    
    # Step 1: Create sample data if PDFs don't exist
    if not os.path.exists(PDF_DIR) or not os.listdir(PDF_DIR):
        print("Creating sample documents...")
        create_sample_pdfs()
    
    # Step 2: Process PDFs and create chunks
    print("Processing documents and creating chunks...")
    processor = PDFProcessor()
    
    # Process text files (simulating PDFs)
    total_chunks = 0
    for filename in os.listdir(PDF_DIR):
        if filename.endswith('.txt'):
            filepath = os.path.join(PDF_DIR, filename)
            # Simulate PDF processing by reading text file
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Create chunks directly
            chunks = processor.split_into_chunks(text, filename.replace('.txt', '.pdf'))
            processor.store_chunks(chunks)
            total_chunks += len(chunks)
            print(f"Processed {filename}: {len(chunks)} chunks")
    
    # Update FTS index
    processor._update_fts_index()
    
    print(f"Total chunks created: {total_chunks}")
    
    # Step 3: Generate embeddings and create vector index
    print("Generating embeddings and creating vector index...")
    vector_search = VectorSearch()
    vector_search.create_index()
    
    print("Setup complete!")
    print(f"Database: {DATABASE_PATH}")
    print(f"Vector index: {FAISS_INDEX_PATH}")
    print(f"Total chunks: {total_chunks}")


if __name__ == "__main__":
    setup_system()
