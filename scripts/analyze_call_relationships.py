#!/usr/bin/env python3
"""
Analyze interprocedural call relationships between files in vulnerabilities.

This module detects if methods in one file are called by another file in
2-file interprocedural vulnerabilities, helping identify true cross-file
dependencies vs. independent changes.
"""

import json
import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CallRelationship:
    """Represents a potential call relationship between files."""
    cve_id: str
    file1_path: str
    file2_path: str
    file1_language: str
    file2_language: str
    has_import: bool
    import_patterns: List[str]
    method_calls: List[Dict]
    confidence: str  # 'high', 'medium', 'low'


class InterproceduralCallAnalyzer:
    """Analyze call relationships in interprocedural vulnerabilities."""
    
    # Language-specific import patterns
    IMPORT_PATTERNS = {
        'python': [
            r'from\s+[\w.]+\s+import\s+',
            r'import\s+[\w.]+',
        ],
        'java': [
            r'import\s+[\w.]+;',
        ],
        'javascript': [
            r'import\s+.*\s+from\s+["\']',
            r'require\s*\(["\']',
        ],
        'typescript': [
            r'import\s+.*\s+from\s+["\']',
            r'require\s*\(["\']',
        ],
        'c': [
            r'#include\s+[<"][\w./]+[>"]',
        ],
        'c++': [
            r'#include\s+[<"][\w./]+[>"]',
        ],
        'ruby': [
            r'require\s+["\'][\w./]+["\']',
            r'require_relative\s+["\']',
        ],
        'go': [
            r'import\s+\(',
            r'import\s+"[\w./]+"',
        ],
        'php': [
            r'require\s+["\']',
            r'require_once\s+["\']',
            r'include\s+["\']',
            r'include_once\s+["\']',
            r'use\s+[\w\\]+;',
        ],
    }
    
    def __init__(self, vulnerabilities_file: str = "interprocedural_vulnerabilities.json"):
        """
        Initialize analyzer with vulnerability data.
        
        Args:
            vulnerabilities_file: Path to JSON file with interprocedural vulnerabilities
        """
        with open(vulnerabilities_file, 'r') as f:
            self.vulnerabilities = json.load(f)
        print(f"✓ Loaded {len(self.vulnerabilities)} interprocedural vulnerabilities")
    
    def get_filename_without_extension(self, filepath: str) -> str:
        """Extract filename without extension from path."""
        if not filepath:
            return ""
        filename = filepath.split('/')[-1]
        return '.'.join(filename.split('.')[:-1]) if '.' in filename else filename
    
    def check_import_relationship(
        self, 
        file1_code: str, 
        file2_code: str,
        file1_path: str,
        file2_path: str,
        language: str
    ) -> Tuple[bool, List[str]]:
        """
        Check if file1 imports/includes file2 or vice versa.
        
        Args:
            file1_code: Source code of first file
            file2_code: Source code of second file
            file1_path: Path of first file
            file2_path: Path of second file
            language: Programming language
            
        Returns:
            Tuple of (has_import, list of import statements found)
        """
        if not file1_code or not file2_code:
            return False, []
        
        if not language:
            return False, []
        
        patterns = self.IMPORT_PATTERNS.get(language.lower(), [])
        if not patterns:
            return False, []
        
        # Get base filenames for matching
        file1_name = self.get_filename_without_extension(file1_path)
        file2_name = self.get_filename_without_extension(file2_path)
        
        found_imports = []
        
        # Check if file1 imports file2
        for pattern in patterns:
            matches_in_file1 = re.findall(pattern, file1_code, re.IGNORECASE)
            for match in matches_in_file1:
                # Check if import mentions file2's name
                if file2_name and file2_name.lower() in match.lower():
                    found_imports.append(f"File1→File2: {match.strip()}")
            
            # Check if file2 imports file1
            matches_in_file2 = re.findall(pattern, file2_code, re.IGNORECASE)
            for match in matches_in_file2:
                # Check if import mentions file1's name
                if file1_name and file1_name.lower() in match.lower():
                    found_imports.append(f"File2→File1: {match.strip()}")
        
        return len(found_imports) > 0, found_imports
    
    def extract_method_names(self, code: str, language: str) -> Set[str]:
        """
        Extract method/function names from code.
        
        Args:
            code: Source code
            language: Programming language
            
        Returns:
            Set of method names
        """
        if not code:
            return set()
        
        if not language:
            return set()
        
        methods = set()
        
        # Language-specific method definition patterns
        patterns = {
            'python': r'def\s+(\w+)\s*\(',
            'java': r'(?:public|private|protected|static|\s)+\w+\s+(\w+)\s*\(',
            'javascript': r'(?:function\s+(\w+)|(\w+)\s*:\s*function|\bconst\s+(\w+)\s*=\s*(?:async\s*)?\()',
            'typescript': r'(?:function\s+(\w+)|(\w+)\s*:\s*function|\bconst\s+(\w+)\s*=\s*(?:async\s*)?\()',
            'c': r'\w+\s+(\w+)\s*\([^)]*\)\s*\{',
            'c++': r'\w+\s+(\w+)\s*\([^)]*\)\s*\{',
            'ruby': r'def\s+(\w+)',
            'go': r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(',
            'php': r'function\s+(\w+)\s*\(',
        }
        
        pattern = patterns.get(language.lower())
        if pattern:
            matches = re.findall(pattern, code, re.MULTILINE)
            # Flatten tuples from patterns with multiple groups
            for match in matches:
                if isinstance(match, tuple):
                    methods.update(m for m in match if m)
                else:
                    methods.add(match)
        
        return methods
    
    def find_method_calls(
        self,
        caller_code: str,
        callee_methods: Set[str],
        language: str
    ) -> List[Dict]:
        """
        Find calls to methods in caller code.
        
        Args:
            caller_code: Source code that may call methods
            callee_methods: Set of method names to look for
            language: Programming language
            
        Returns:
            List of found method calls with context
        """
        if not caller_code or not callee_methods:
            return []
        
        if not language:
            return []
        
        found_calls = []
        
        for method in callee_methods:
            # Look for method invocations: method(), obj.method(), obj->method()
            patterns = [
                rf'\b{re.escape(method)}\s*\(',  # Direct call
                rf'\.\s*{re.escape(method)}\s*\(',  # Method call on object (Java, Python, JS)
                rf'->\s*{re.escape(method)}\s*\(',  # Pointer method call (C++, PHP)
                rf'::\s*{re.escape(method)}\s*\(',  # Static method call (C++, PHP)
            ]
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, caller_code))
                for match in matches:
                    # Get some context around the call
                    start = max(0, match.start() - 50)
                    end = min(len(caller_code), match.end() + 50)
                    context = caller_code[start:end].strip()
                    
                    found_calls.append({
                        'method': method,
                        'pattern': pattern,
                        'context': context,
                        'position': match.start()
                    })
        
        return found_calls
    
    def analyze_call_relationships(self) -> List[CallRelationship]:
        """
        Analyze call relationships in all interprocedural vulnerabilities.
        
        Returns:
            List of CallRelationship objects
        """
        relationships = []
        
        for vuln in self.vulnerabilities:
            if len(vuln['files']) != 2:
                continue
            
            file1 = vuln['files'][0]
            file2 = vuln['files'][1]
            
            # Get code and metadata
            file1_code = file1.get('code_before', '')
            file2_code = file2.get('code_before', '')
            file1_path = file1.get('filename', '')
            file2_path = file2.get('filename', '')
            file1_lang = file1.get('language', 'Unknown')
            file2_lang = file2.get('language', 'Unknown')
            
            # Skip if no code
            if not file1_code or not file2_code:
                continue
            
            # Check for imports
            has_import, import_patterns = self.check_import_relationship(
                file1_code, file2_code, file1_path, file2_path,
                file1_lang if file1_lang == file2_lang else file1_lang
            )
            
            # Extract methods from each file
            file1_methods = self.extract_method_names(file1_code, file1_lang)
            file2_methods = self.extract_method_names(file2_code, file2_lang)
            
            # Find calls from file1 to file2's methods
            calls_1_to_2 = self.find_method_calls(file1_code, file2_methods, file1_lang)
            # Find calls from file2 to file1's methods
            calls_2_to_1 = self.find_method_calls(file2_code, file1_methods, file2_lang)
            
            all_calls = calls_1_to_2 + calls_2_to_1
            
            # Determine confidence level
            confidence = 'low'
            if has_import and len(all_calls) > 0:
                confidence = 'high'
            elif has_import or len(all_calls) > 0:
                confidence = 'medium'
            
            relationship = CallRelationship(
                cve_id=vuln['cve_id'],
                file1_path=file1_path,
                file2_path=file2_path,
                file1_language=file1_lang,
                file2_language=file2_lang,
                has_import=has_import,
                import_patterns=import_patterns,
                method_calls=all_calls,
                confidence=confidence
            )
            
            relationships.append(relationship)
        
        return relationships
    
    def filter_by_confidence(
        self,
        relationships: List[CallRelationship],
        min_confidence: str = 'medium'
    ) -> List[CallRelationship]:
        """
        Filter relationships by confidence level.
        
        Args:
            relationships: List of call relationships
            min_confidence: Minimum confidence level ('low', 'medium', 'high')
            
        Returns:
            Filtered list
        """
        confidence_order = {'low': 0, 'medium': 1, 'high': 2}
        min_level = confidence_order.get(min_confidence, 1)
        
        return [
            rel for rel in relationships
            if confidence_order.get(rel.confidence, 0) >= min_level
        ]
    
    def get_statistics(self, relationships: List[CallRelationship]) -> Dict:
        """
        Get statistics about call relationships.
        
        Args:
            relationships: List of call relationships
            
        Returns:
            Dictionary with statistics
        """
        total = len(relationships)
        
        has_imports = sum(1 for r in relationships if r.has_import)
        has_calls = sum(1 for r in relationships if len(r.method_calls) > 0)
        has_both = sum(1 for r in relationships if r.has_import and len(r.method_calls) > 0)
        
        by_confidence = {
            'high': sum(1 for r in relationships if r.confidence == 'high'),
            'medium': sum(1 for r in relationships if r.confidence == 'medium'),
            'low': sum(1 for r in relationships if r.confidence == 'low'),
        }
        
        same_language = sum(1 for r in relationships if r.file1_language == r.file2_language)
        
        return {
            'total_analyzed': total,
            'with_imports': has_imports,
            'with_method_calls': has_calls,
            'with_both': has_both,
            'by_confidence': by_confidence,
            'same_language': same_language,
            'cross_language': total - same_language,
        }
    
    def save_results(self, relationships: List[CallRelationship], filename: str):
        """Save analysis results to JSON file."""
        data = []
        for rel in relationships:
            data.append({
                'cve_id': rel.cve_id,
                'file1_path': rel.file1_path,
                'file2_path': rel.file2_path,
                'file1_language': rel.file1_language,
                'file2_language': rel.file2_language,
                'has_import': rel.has_import,
                'import_patterns': rel.import_patterns,
                'method_calls': rel.method_calls,
                'confidence': rel.confidence,
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved {len(data)} call relationships to {filename}")


def main():
    """Main execution to analyze interprocedural call relationships."""
    print("\n" + "="*80)
    print("INTERPROCEDURAL CALL RELATIONSHIP ANALYSIS")
    print("="*80)
    print("\nAnalyzing whether methods in one file are called by another file...")
    
    # Initialize analyzer
    analyzer = InterproceduralCallAnalyzer()
    
    # Analyze relationships
    print("\n" + "="*80)
    print("ANALYZING CALL RELATIONSHIPS")
    print("="*80)
    
    relationships = analyzer.analyze_call_relationships()
    
    print(f"\n✓ Analyzed {len(relationships)} file pairs")
    
    # Get statistics
    stats = analyzer.get_statistics(relationships)
    
    print("\n📊 Overall Statistics:")
    print(f"  Total file pairs analyzed: {stats['total_analyzed']}")
    print(f"  With import statements: {stats['with_imports']}")
    print(f"  With method calls detected: {stats['with_method_calls']}")
    print(f"  With both imports and calls: {stats['with_both']}")
    print(f"\nBy confidence:")
    print(f"  High confidence: {stats['by_confidence']['high']}")
    print(f"  Medium confidence: {stats['by_confidence']['medium']}")
    print(f"  Low confidence: {stats['by_confidence']['low']}")
    print(f"\nLanguage mixing:")
    print(f"  Same language: {stats['same_language']}")
    print(f"  Cross-language: {stats['cross_language']}")
    
    # Show high-confidence examples
    high_confidence = analyzer.filter_by_confidence(relationships, 'high')
    
    if high_confidence:
        print("\n" + "="*80)
        print(f"HIGH-CONFIDENCE CALL RELATIONSHIPS ({len(high_confidence)} found)")
        print("="*80)
        
        for rel in high_confidence[:5]:
            print(f"\nCVE: {rel.cve_id}")
            print(f"  File 1: {rel.file1_path} ({rel.file1_language})")
            print(f"  File 2: {rel.file2_path} ({rel.file2_language})")
            if rel.import_patterns:
                print(f"  Imports:")
                for imp in rel.import_patterns:
                    print(f"    - {imp}")
            if rel.method_calls:
                print(f"  Method calls ({len(rel.method_calls)}):")
                for call in rel.method_calls[:3]:
                    print(f"    - {call['method']}()")
                    print(f"      Context: {call['context'][:100]}...")
    
    # Save all results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    analyzer.save_results(relationships, "call_relationships_all.json")
    analyzer.save_results(
        analyzer.filter_by_confidence(relationships, 'high'),
        "call_relationships_high_confidence.json"
    )
    analyzer.save_results(
        analyzer.filter_by_confidence(relationships, 'medium'),
        "call_relationships_medium_plus.json"
    )
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - call_relationships_all.json")
    print("  - call_relationships_high_confidence.json")
    print("  - call_relationships_medium_plus.json")


if __name__ == "__main__":
    main()
