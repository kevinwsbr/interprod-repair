#!/usr/bin/env python3
"""
Extract interprocedural vulnerabilities that span across multiple files.

This module identifies CVE vulnerabilities where the fix involves changes to
multiple files (n=2 for now), suggesting the vulnerability spans across file
boundaries. Test files and non-code files (text, markdown, documentation) are
automatically excluded to focus on actual code changes.
"""

import psycopg2
import json
import csv
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict


@dataclass
class InterproceduralVulnerability:
    """Data class for interprocedural vulnerability involving multiple files."""
    cve_id: str
    cve_description: str
    cwe_id: Optional[str]
    cwe_name: Optional[str]
    severity: Optional[float]
    commit_id: str
    commit_message: str
    commit_date: str
    repository: str
    file_count: int
    files: List[Dict] = field(default_factory=list)


class InterproceduralVulnerabilityExtractor:
    """Extract vulnerabilities that span across multiple files."""
    
    # Common test file patterns to exclude
    TEST_PATTERNS = [
        r'test[_/]',           # test_ prefix or test/ directory
        r'_test\.',            # _test.py, _test.java suffix
        r'tests?/',            # test/ or tests/ directory
        r'spec[_/]',           # spec_ prefix or spec/ directory
        r'_spec\.',            # _spec.rb suffix
        r'Test\.java$',        # Java test classes
        r'Tests?\.py$',        # Python test classes
        r'\.test\.',           # .test.js, .test.ts
        r'\.spec\.',           # .spec.js, .spec.ts
        r'__tests?__/',        # __test__/ or __tests__/ directory
        r'fixture',            # fixture files
        r'mock',               # mock files
        r'/examples?/',        # example directories
        r'/samples?/',         # sample directories
        r'benchmark',          # benchmark files
    ]
    
    # Non-code file patterns to exclude (documentation, text files)
    NON_CODE_PATTERNS = [
        r'\.txt$',             # Plain text files
        r'\.md$',              # Markdown files
        r'\.markdown$',        # Markdown files (alternative extension)
        r'\.rst$',             # reStructuredText
        r'\.adoc$',            # AsciiDoc
        r'\.org$',             # Org mode files
        r'\.tex$',             # LaTeX
        r'\.doc$',             # Word documents
        r'\.docx$',            # Word documents
        r'\.pdf$',             # PDF files
        r'\.rtf$',             # Rich text format
        r'^README',            # README files (any extension)
        r'^CHANGELOG',         # CHANGELOG files
        r'^CHANGES',           # CHANGES files
        r'^LICENSE',           # LICENSE files
        r'^COPYING',           # COPYING files
        r'^AUTHORS',           # AUTHORS files
        r'^CONTRIBUTORS',      # CONTRIBUTORS files
        r'^HISTORY',           # HISTORY files
        r'^NEWS',              # NEWS files
        r'\.log$',             # Log files
    ]
    
    def __init__(self, dbname: str = "tcc", user: str = "postgres", 
                 password: str = "", host: str = "localhost", port: str = "5432"):
        """
        Initialize database connection.
        
        Args:
            dbname: Database name
            user: Database user
            password: Database password
            host: Database host
            port: Database port
        """
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.cursor = self.conn.cursor()
        print(f"✓ Connected to database '{dbname}'")
        
        # Compile test patterns for efficiency
        self.test_regex = re.compile('|'.join(self.TEST_PATTERNS), re.IGNORECASE)
        # Compile non-code patterns for efficiency
        self.non_code_regex = re.compile('|'.join(self.NON_CODE_PATTERNS), re.IGNORECASE)
    
    def is_test_file(self, filepath: str) -> bool:
        """
        Check if a file path appears to be a test file.
        
        Args:
            filepath: File path to check
            
        Returns:
            True if file appears to be a test file, False otherwise
        """
        if not filepath:
            return False
        return bool(self.test_regex.search(filepath))
    
    def is_non_code_file(self, filepath: str) -> bool:
        """
        Check if a file path appears to be a non-code file (documentation, text, etc.).
        
        Args:
            filepath: File path to check
            
        Returns:
            True if file appears to be a non-code file, False otherwise
        """
        if not filepath:
            return False
        # Extract just the filename for pattern matching
        filename = filepath.split('/')[-1] if '/' in filepath else filepath
        return bool(self.non_code_regex.search(filename))
    
    def extract_interprocedural_vulnerabilities(
        self,
        num_files: int = 2,
        min_severity: Optional[float] = None,
        language: Optional[str] = None,
        cwe_ids: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[InterproceduralVulnerability]:
        """
        Extract vulnerabilities that span across multiple files.
        
        This identifies CVEs where the fixing commit modified exactly N files
        (excluding test files), suggesting interprocedural vulnerability patterns.
        
        Args:
            num_files: Number of files involved (default 2 for pairwise)
            min_severity: Minimum CVSS v3.1 base score
            language: Filter by programming language
            cwe_ids: List of CWE IDs to filter
            limit: Maximum number of CVEs to return
            
        Returns:
            List of InterproceduralVulnerability objects
        """
        # First, find CVEs with commits that modified exactly N files (non-test)
        # We need to count files per commit excluding test files
        query = """
            WITH commit_files AS (
                SELECT 
                    fx.cve_id,
                    fx.commit_id,
                    co.message as commit_message,
                    co.date as commit_date,
                    r.name as repository,
                    ARRAY_AGG(
                        json_build_object(
                            'file_id', f.id,
                            'filename', f.filename,
                            'old_path', f.old_path,
                            'new_path', f.new_path,
                            'language', f.language,
                            'code_before', f.code_before,
                            'code_after', f.code_after,
                            'diff', f.diff,
                            'change_type', f.change_type,
                            'lines_added', f.lines_added,
                            'lines_removed', f.lines_removed,
                            'nloc', f.nloc,
                            'complexity', f.complexity
                        ) ORDER BY f.filename
                    ) as files
                FROM fixes fx
                JOIN commits co ON fx.commit_id = co.id
                JOIN files f ON co.id = f.commit_id
                JOIN repositories r ON co.repository_url = r.url
                WHERE f.code_before IS NOT NULL
        """
        
        params = []
        
        if language is not None:
            query += " AND f.language = %s"
            params.append(language)
        
        query += """
                GROUP BY fx.cve_id, fx.commit_id, co.message, co.date, r.name
            )
            SELECT 
                c.id as cve_id,
                c.description as cve_description,
                cw.id as cwe_id,
                cw.name as cwe_name,
                c.v31_base_score as severity,
                cf.commit_id,
                cf.commit_message,
                cf.commit_date,
                cf.repository,
                cf.files
            FROM commit_files cf
            JOIN cves c ON cf.cve_id = c.id
            LEFT JOIN classifications cl ON c.id = cl.cve_id
            LEFT JOIN cwes cw ON cl.cwe_id = cw.id
            WHERE 1=1
        """
        
        if min_severity is not None:
            query += " AND c.v31_base_score >= %s"
            params.append(min_severity)
        
        if cwe_ids is not None and len(cwe_ids) > 0:
            query += " AND cw.id = ANY(%s)"
            params.append(cwe_ids)
        
        query += " ORDER BY c.v31_base_score DESC NULLS LAST, c.id"
        
        if limit is not None:
            query += " LIMIT %s"
            params.append(limit)
        
        self.cursor.execute(query, params)
        
        results = []
        filtered_counts = {
            'total_examined': 0,
            'excluded_test_files': 0,
            'excluded_non_code_files': 0,
            'wrong_file_count': 0,
            'included': 0
        }
        
        for row in self.cursor.fetchall():
            filtered_counts['total_examined'] += 1
            
            files = row[9]  # JSON array of files
            
            # Filter out test files and non-code files
            code_files = []
            for file_info in files:
                # Check both filename and paths for test patterns
                filename = file_info.get('filename', '')
                old_path = file_info.get('old_path', '')
                new_path = file_info.get('new_path', '')
                
                paths_to_check = [filename, old_path, new_path]
                is_test = any(self.is_test_file(path) for path in paths_to_check if path)
                is_non_code = any(self.is_non_code_file(path) for path in paths_to_check if path)
                
                if is_test:
                    filtered_counts['excluded_test_files'] += 1
                elif is_non_code:
                    filtered_counts['excluded_non_code_files'] += 1
                else:
                    code_files.append(file_info)
            
            # Only include if exactly N code files
            if len(code_files) != num_files:
                filtered_counts['wrong_file_count'] += 1
                continue
            
            filtered_counts['included'] += 1
            
            vuln = InterproceduralVulnerability(
                cve_id=row[0],
                cve_description=row[1],
                cwe_id=row[2],
                cwe_name=row[3],
                severity=row[4],
                commit_id=row[5],
                commit_message=row[6],
                commit_date=str(row[7]) if row[7] else None,
                repository=row[8],
                file_count=len(code_files),
                files=code_files
            )
            results.append(vuln)
        
        print(f"\n📊 Filtering Statistics:")
        print(f"  Total CVEs examined: {filtered_counts['total_examined']}")
        print(f"  Test files excluded: {filtered_counts['excluded_test_files']}")
        print(f"  Non-code files excluded: {filtered_counts['excluded_non_code_files']}")
        print(f"  Wrong file count: {filtered_counts['wrong_file_count']}")
        print(f"  ✓ Interprocedural vulnerabilities found: {filtered_counts['included']}")
        
        return results
    
    def analyze_file_pairs(
        self, 
        vulnerabilities: List[InterproceduralVulnerability]
    ) -> Dict:
        """
        Analyze patterns in file pairs involved in interprocedural vulnerabilities.
        
        Args:
            vulnerabilities: List of interprocedural vulnerabilities
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'total_vulnerabilities': len(vulnerabilities),
            'language_pairs': defaultdict(int),
            'directory_patterns': defaultdict(int),
            'file_type_pairs': defaultdict(int),
            'same_language_count': 0,
            'cross_language_count': 0
        }
        
        for vuln in vulnerabilities:
            if len(vuln.files) != 2:
                continue
            
            file1, file2 = vuln.files[0], vuln.files[1]
            
            # Language analysis
            lang1 = file1.get('language') or 'Unknown'
            lang2 = file2.get('language') or 'Unknown'
            
            lang_pair = tuple(sorted([lang1, lang2]))
            analysis['language_pairs'][lang_pair] += 1
            
            if lang1 == lang2:
                analysis['same_language_count'] += 1
            else:
                analysis['cross_language_count'] += 1
            
            # Directory pattern analysis
            path1 = file1.get('filename') or file1.get('new_path') or ''
            path2 = file2.get('filename') or file2.get('new_path') or ''
            
            dir1 = '/'.join(path1.split('/')[:-1]) if '/' in path1 else 'root'
            dir2 = '/'.join(path2.split('/')[:-1]) if '/' in path2 else 'root'
            
            if dir1 == dir2:
                analysis['directory_patterns']['same_directory'] += 1
            else:
                analysis['directory_patterns']['different_directories'] += 1
            
            # File type pairs (extension)
            ext1 = '.' + path1.split('.')[-1] if '.' in path1 else 'no_ext'
            ext2 = '.' + path2.split('.')[-1] if '.' in path2 else 'no_ext'
            
            ext_pair = tuple(sorted([ext1, ext2]))
            analysis['file_type_pairs'][ext_pair] += 1
        
        # Convert defaultdicts to regular dicts for JSON serialization
        # Convert tuple keys to strings for JSON compatibility
        analysis['language_pairs'] = {
            f"{k[0]} + {k[1]}": v for k, v in analysis['language_pairs'].items()
        }
        analysis['directory_patterns'] = dict(analysis['directory_patterns'])
        analysis['file_type_pairs'] = {
            f"{k[0]} + {k[1]}": v for k, v in analysis['file_type_pairs'].items()
        }
        
        return analysis
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about interprocedural vulnerabilities in the database.
        
        Returns:
            Dictionary containing statistics
        """
        stats = {}
        
        # Count CVEs by number of files modified (excluding test files)
        # This is complex so we'll do it in Python for now
        query = """
            SELECT 
                fx.cve_id,
                f.filename,
                f.old_path,
                f.new_path
            FROM fixes fx
            JOIN commits co ON fx.commit_id = co.id
            JOIN files f ON co.id = f.commit_id
            WHERE f.code_before IS NOT NULL
            ORDER BY fx.cve_id
        """
        
        self.cursor.execute(query)
        
        cve_files = defaultdict(list)
        for row in self.cursor.fetchall():
            cve_id = row[0]
            paths = [p for p in row[1:] if p]
            # Exclude test files and non-code files
            is_test = any(self.is_test_file(p) for p in paths)
            is_non_code = any(self.is_non_code_file(p) for p in paths)
            if not is_test and not is_non_code:
                cve_files[cve_id].append(paths)
        
        file_count_distribution = defaultdict(int)
        for cve_id, files in cve_files.items():
            file_count_distribution[len(files)] += 1
        
        stats['file_count_distribution'] = dict(sorted(file_count_distribution.items()))
        stats['total_cves_analyzed'] = len(cve_files)
        
        # Multi-file vulnerabilities
        multi_file_cves = sum(1 for files in cve_files.values() if len(files) >= 2)
        stats['multi_file_vulnerabilities'] = multi_file_cves
        stats['multi_file_percentage'] = round(
            100.0 * multi_file_cves / len(cve_files), 2
        ) if len(cve_files) > 0 else 0
        
        return stats
    
    def save_to_json(self, data: List, filename: str):
        """
        Save extracted data to JSON file.
        
        Args:
            data: List of data objects to save
            filename: Output filename
        """
        # Convert dataclass objects to dicts
        if data and hasattr(data[0], '__dataclass_fields__'):
            data = [asdict(item) for item in data]
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"✓ Saved {len(data)} vulnerabilities to {filename}")
    
    def save_to_csv(self, data: List[InterproceduralVulnerability], filename: str):
        """
        Save extracted data to CSV file (flattened format).
        
        Args:
            data: List of InterproceduralVulnerability objects
            filename: Output filename
        """
        if not data:
            print("No data to save")
            return
        
        # Flatten the data for CSV (one row per vulnerability with file info)
        rows = []
        for vuln in data:
            file_info = []
            for i, file_data in enumerate(vuln.files, 1):
                file_info.append({
                    f'file{i}_name': file_data.get('filename'),
                    f'file{i}_language': file_data.get('language'),
                    f'file{i}_lines_added': file_data.get('lines_added'),
                    f'file{i}_lines_removed': file_data.get('lines_removed'),
                    f'file{i}_complexity': file_data.get('complexity'),
                })
            
            # Merge all file info into one row
            row = {
                'cve_id': vuln.cve_id,
                'severity': vuln.severity,
                'cwe_id': vuln.cwe_id,
                'cwe_name': vuln.cwe_name,
                'file_count': vuln.file_count,
                'commit_id': vuln.commit_id,
                'repository': vuln.repository,
                'commit_date': vuln.commit_date,
            }
            
            # Add file-specific columns
            for file_dict in file_info:
                row.update(file_dict)
            
            rows.append(row)
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        
        print(f"✓ Saved {len(rows)} vulnerabilities to {filename}")
    
    def close(self):
        """Close database connection."""
        self.cursor.close()
        self.conn.close()
        print("✓ Database connection closed")


def main():
    """Main execution function demonstrating interprocedural vulnerability extraction."""
    extractor = InterproceduralVulnerabilityExtractor()
    
    print("\n" + "="*80)
    print("INTERPROCEDURAL VULNERABILITY ANALYSIS")
    print("="*80)
    print("\nExtracting vulnerabilities that span across multiple files...")
    print("Test files and non-code files (text, markdown, docs) are automatically excluded.")
    
    # Get overall statistics
    print("\n" + "="*80)
    print("DATABASE STATISTICS")
    print("="*80)
    stats = extractor.get_statistics()
    print(f"\nTotal CVEs with code files: {stats['total_cves_analyzed']}")
    print(f"Multi-file vulnerabilities (≥2 files): {stats['multi_file_vulnerabilities']} "
          f"({stats['multi_file_percentage']}%)")
    
    print("\nDistribution by file count:")
    for count, num_cves in sorted(stats['file_count_distribution'].items())[:10]:
        print(f"  {count} file(s): {num_cves} CVEs")
    
    # Extract 2-file interprocedural vulnerabilities
    print("\n" + "="*80)
    print("EXTRACTING 2-FILE INTERPROCEDURAL VULNERABILITIES")
    print("="*80)
    
    vulnerabilities = extractor.extract_interprocedural_vulnerabilities(
        num_files=2,
        limit=None  # Get all 2-file vulnerabilities, not just top 100
    )
    
    if vulnerabilities:
        print(f"\n✓ Found {len(vulnerabilities)} interprocedural vulnerabilities")
        
        # Show examples
        print("\n📋 Examples (first 3):")
        for vuln in vulnerabilities[:3]:
            print(f"\n  CVE: {vuln.cve_id}")
            print(f"  Severity: {vuln.severity}")
            print(f"  CWE: {vuln.cwe_id} - {vuln.cwe_name}")
            print(f"  Repository: {vuln.repository}")
            print(f"  Files involved:")
            for i, file_data in enumerate(vuln.files, 1):
                print(f"    {i}. {file_data['filename']} ({file_data['language']})")
                print(f"       +{file_data['lines_added']}/-{file_data['lines_removed']} lines")
        
        # Analyze patterns
        print("\n" + "="*80)
        print("PATTERN ANALYSIS")
        print("="*80)
        
        analysis = extractor.analyze_file_pairs(vulnerabilities)
        
        print(f"\nLanguage patterns:")
        print(f"  Same language: {analysis['same_language_count']}")
        print(f"  Cross-language: {analysis['cross_language_count']}")
        
        print(f"\nTop language pairs:")
        sorted_lang_pairs = sorted(
            analysis['language_pairs'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        for lang_pair_str, count in sorted_lang_pairs[:10]:
            print(f"  {lang_pair_str}: {count}")
        
        print(f"\nDirectory patterns:")
        for pattern, count in analysis['directory_patterns'].items():
            print(f"  {pattern}: {count}")
        
        print(f"\nTop file type pairs:")
        sorted_ext_pairs = sorted(
            analysis['file_type_pairs'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for ext_pair_str, count in sorted_ext_pairs[:10]:
            print(f"  {ext_pair_str}: {count}")
        
        # Save results
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        extractor.save_to_json(vulnerabilities, "interprocedural_vulnerabilities.json")
        extractor.save_to_csv(vulnerabilities, "interprocedural_vulnerabilities.csv")
        
        # Save analysis
        with open("interprocedural_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"✓ Saved analysis to interprocedural_analysis.json")
        
        # Extract high-severity interprocedural vulnerabilities
        print("\n" + "="*80)
        print("HIGH-SEVERITY INTERPROCEDURAL VULNERABILITIES (CVSS ≥ 7.0)")
        print("="*80)
        
        high_severity = extractor.extract_interprocedural_vulnerabilities(
            num_files=2,
            min_severity=7.0,
            limit=None  # Get all high-severity 2-file vulnerabilities
        )
        
        if high_severity:
            print(f"\n✓ Found {len(high_severity)} high-severity interprocedural vulnerabilities")
            extractor.save_to_json(high_severity, "interprocedural_high_severity.json")
        else:
            print("\nNo high-severity interprocedural vulnerabilities found")
    
    else:
        print("\n⚠ No interprocedural vulnerabilities found matching the criteria")
    
    extractor.close()
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - interprocedural_vulnerabilities.json")
    print("  - interprocedural_vulnerabilities.csv")
    print("  - interprocedural_analysis.json")
    print("  - interprocedural_high_severity.json")


if __name__ == "__main__":
    main()
