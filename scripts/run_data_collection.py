#!/usr/bin/env python3
"""
DBS Data Collection Orchestrator
Phase 2: Data Collection - Main Orchestration Script

This script orchestrates the entire data collection pipeline including:
- Web scraping
- PDF processing
- Data cleaning and preprocessing
- Quality assurance
- Knowledge base construction

Author: DBS Chatbot Project
Date: October 2024
"""

import asyncio
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataCollectionPipeline:
    """Main orchestrator for data collection pipeline"""
    
    def __init__(self, config_file: str = "data_collection_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.pipeline_stats = {
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'total_duration': 0,
            'steps_completed': [],
            'steps_failed': [],
            'overall_success': False,
            'data_collected': {
                'web_pages': 0,
                'pdf_documents': 0,
                'total_chunks': 0,
                'knowledge_base_items': 0
            }
        }
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "web_scraping": {
                "enabled": True,
                "base_url": "https://www.dbs.ie",
                "delay": 1.0,
                "max_pages": 100
            },
            "pdf_processing": {
                "enabled": True,
                "input_dir": "data/raw/pdfs",
                "max_file_size": 10485760  # 10MB
            },
            "data_cleaning": {
                "enabled": True,
                "min_quality_score": 0.3,
                "remove_duplicates": True
            },
            "quality_assurance": {
                "enabled": True,
                "min_quality_score": 0.5,
                "validate_urls": True
            },
            "knowledge_base": {
                "enabled": True,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "collection_name": "dbs_documents"
            }
        }
        
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                logger.warning(f"Error loading config file: {e}. Using defaults.")
                return default_config
        else:
            # Create default config file
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config file: {self.config_file}")
            return default_config
    
    def run_script(self, script_path: str, description: str) -> bool:
        """Run a Python script and return success status"""
        try:
            logger.info(f"Starting {description}...")
            start_time = time.time()
            
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ {description} completed successfully in {duration:.2f} seconds")
                self.pipeline_stats['steps_completed'].append({
                    'step': description,
                    'duration': duration,
                    'status': 'success'
                })
                return True
            else:
                logger.error(f"❌ {description} failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                self.pipeline_stats['steps_failed'].append({
                    'step': description,
                    'duration': duration,
                    'status': 'failed',
                    'error': result.stderr
                })
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {description} timed out after 1 hour")
            self.pipeline_stats['steps_failed'].append({
                'step': description,
                'duration': 3600,
                'status': 'timeout',
                'error': 'Process timed out after 1 hour'
            })
            return False
        except Exception as e:
            logger.error(f"❌ Error running {description}: {str(e)}")
            self.pipeline_stats['steps_failed'].append({
                'step': description,
                'duration': 0,
                'status': 'error',
                'error': str(e)
            })
            return False
    
    async def run_web_scraping(self) -> bool:
        """Run web scraping step"""
        if not self.config['web_scraping']['enabled']:
            logger.info("Web scraping disabled in config")
            return True
        
        return self.run_script(
            "scripts/scrape_data.py",
            "Web Scraping"
        )
    
    def run_pdf_processing(self) -> bool:
        """Run PDF processing step"""
        if not self.config['pdf_processing']['enabled']:
            logger.info("PDF processing disabled in config")
            return True
        
        return self.run_script(
            "scripts/process_pdfs.py",
            "PDF Processing"
        )
    
    def run_data_cleaning(self) -> bool:
        """Run data cleaning step"""
        if not self.config['data_cleaning']['enabled']:
            logger.info("Data cleaning disabled in config")
            return True
        
        return self.run_script(
            "scripts/clean_and_preprocess.py",
            "Data Cleaning and Preprocessing"
        )
    
    def run_quality_assurance(self) -> bool:
        """Run quality assurance step"""
        if not self.config['quality_assurance']['enabled']:
            logger.info("Quality assurance disabled in config")
            return True
        
        return self.run_script(
            "scripts/quality_assurance.py",
            "Quality Assurance"
        )
    
    def run_knowledge_base_construction(self) -> bool:
        """Run knowledge base construction step"""
        if not self.config['knowledge_base']['enabled']:
            logger.info("Knowledge base construction disabled in config")
            return True
        
        return self.run_script(
            "scripts/build_knowledge_base.py",
            "Knowledge Base Construction"
        )
    
    def collect_pipeline_statistics(self) -> Dict[str, Any]:
        """Collect statistics from pipeline execution"""
        stats = {
            'web_scraping': {},
            'pdf_processing': {},
            'data_cleaning': {},
            'quality_assurance': {},
            'knowledge_base': {}
        }
        
        # Collect web scraping stats
        web_report_file = Path("data/raw/scraping_report.json")
        if web_report_file.exists():
            try:
                with open(web_report_file, 'r') as f:
                    stats['web_scraping'] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load web scraping stats: {e}")
        
        # Collect PDF processing stats
        pdf_report_file = Path("data/processed/pdf_processing_report.json")
        if pdf_report_file.exists():
            try:
                with open(pdf_report_file, 'r') as f:
                    stats['pdf_processing'] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load PDF processing stats: {e}")
        
        # Collect data cleaning stats
        cleaning_report_file = Path("data/processed/cleaning_report.json")
        if cleaning_report_file.exists():
            try:
                with open(cleaning_report_file, 'r') as f:
                    stats['data_cleaning'] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load data cleaning stats: {e}")
        
        # Collect quality assurance stats
        qa_report_file = Path("data/processed/web_quality_reports_stats.json")
        if qa_report_file.exists():
            try:
                with open(qa_report_file, 'r') as f:
                    stats['quality_assurance'] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load quality assurance stats: {e}")
        
        # Collect knowledge base stats
        kb_report_file = Path("data/processed/knowledge_base_stats.json")
        if kb_report_file.exists():
            try:
                with open(kb_report_file, 'r') as f:
                    stats['knowledge_base'] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load knowledge base stats: {e}")
        
        return stats
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate final pipeline report"""
        self.pipeline_stats['end_time'] = datetime.now().isoformat()
        
        # Calculate total duration
        start_time = datetime.fromisoformat(self.pipeline_stats['start_time'])
        end_time = datetime.fromisoformat(self.pipeline_stats['end_time'])
        self.pipeline_stats['total_duration'] = (end_time - start_time).total_seconds()
        
        # Collect detailed statistics
        detailed_stats = self.collect_pipeline_statistics()
        self.pipeline_stats['detailed_statistics'] = detailed_stats
        
        # Determine overall success
        self.pipeline_stats['overall_success'] = len(self.pipeline_stats['steps_failed']) == 0
        
        # Update data collected counts
        if 'web_scraping' in detailed_stats:
            self.pipeline_stats['data_collected']['web_pages'] = detailed_stats['web_scraping'].get('total_pages', 0)
        
        if 'pdf_processing' in detailed_stats:
            self.pipeline_stats['data_collected']['pdf_documents'] = detailed_stats['pdf_processing'].get('total_pdfs', 0)
        
        if 'knowledge_base' in detailed_stats:
            self.pipeline_stats['data_collected']['total_chunks'] = detailed_stats['knowledge_base'].get('total_chunks_created', 0)
            self.pipeline_stats['data_collected']['knowledge_base_items'] = detailed_stats['knowledge_base'].get('total_items_processed', 0)
        
        return self.pipeline_stats
    
    def save_final_report(self, report_file: str = "data/processed/pipeline_final_report.json"):
        """Save final pipeline report"""
        report_path = Path(report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.pipeline_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Final pipeline report saved to {report_file}")
    
    async def run_pipeline(self) -> bool:
        """Run the complete data collection pipeline"""
        logger.info("🚀 Starting DBS Data Collection Pipeline...")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        # Step 1: Web Scraping
        web_success = await self.run_web_scraping()
        if not web_success and self.config['web_scraping']['enabled']:
            logger.error("❌ Web scraping failed, continuing with other steps...")
        
        # Step 2: PDF Processing
        pdf_success = self.run_pdf_processing()
        if not pdf_success and self.config['pdf_processing']['enabled']:
            logger.error("❌ PDF processing failed, continuing with other steps...")
        
        # Step 3: Data Cleaning
        cleaning_success = self.run_data_cleaning()
        if not cleaning_success and self.config['data_cleaning']['enabled']:
            logger.error("❌ Data cleaning failed, stopping pipeline...")
            return False
        
        # Step 4: Quality Assurance
        qa_success = self.run_quality_assurance()
        if not qa_success and self.config['quality_assurance']['enabled']:
            logger.error("❌ Quality assurance failed, continuing with knowledge base...")
        
        # Step 5: Knowledge Base Construction
        kb_success = self.run_knowledge_base_construction()
        if not kb_success and self.config['knowledge_base']['enabled']:
            logger.error("❌ Knowledge base construction failed, pipeline incomplete...")
            return False
        
        # Generate final report
        final_report = self.generate_final_report()
        self.save_final_report()
        
        # Log final results
        if final_report['overall_success']:
            logger.info("🎉 Data Collection Pipeline completed successfully!")
        else:
            logger.warning("⚠️ Data Collection Pipeline completed with some failures")
        
        logger.info(f"📊 Pipeline Statistics:")
        logger.info(f"   - Total Duration: {final_report['total_duration']:.2f} seconds")
        logger.info(f"   - Steps Completed: {len(final_report['steps_completed'])}")
        logger.info(f"   - Steps Failed: {len(final_report['steps_failed'])}")
        logger.info(f"   - Web Pages: {final_report['data_collected']['web_pages']}")
        logger.info(f"   - PDF Documents: {final_report['data_collected']['pdf_documents']}")
        logger.info(f"   - Total Chunks: {final_report['data_collected']['total_chunks']}")
        logger.info(f"   - Knowledge Base Items: {final_report['data_collected']['knowledge_base_items']}")
        
        return final_report['overall_success']

async def main():
    """Main function to run the data collection pipeline"""
    # Initialize pipeline
    pipeline = DataCollectionPipeline()
    
    # Run pipeline
    success = await pipeline.run_pipeline()
    
    if success:
        print("\n🎉 Data Collection Pipeline completed successfully!")
        print("📁 Check the 'data/processed/' directory for results")
        print("📊 Check 'data/processed/pipeline_final_report.json' for detailed statistics")
    else:
        print("\n❌ Data Collection Pipeline completed with errors")
        print("📋 Check the logs for details on what went wrong")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
