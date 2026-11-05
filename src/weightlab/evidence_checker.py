"""Deterministic checker for the study's explicit field=value record syntax."""
import re
PATTERNS=[r'Required field:\s*([a-z_]+)',r'Retrieve field \[([a-z_]+)\]',r'requested field is ([a-z_]+)',r'Required=([a-z_]+);',r'For required field ([a-z_]+),']
def check_record(text):
 fields=[m.group(1).lower() for pattern in PATTERNS if (m:=re.search(pattern,text,re.I))]
 if len(set(fields))!=1:raise ValueError('Unsupported or ambiguous required-field syntax')
 field=fields[0];facts=re.findall(r'\b([a-z_]+)=([A-Za-z0-9_-]+)',text)
 values={v for k,v in facts if k==field}
 return field,{'value':next(iter(values)),'ask':None} if len(values)==1 else {'value':None,'ask':field}
def check_update(field,text):
 matches=re.findall(r'Authoritative update:\s*([a-z_]+)=([A-Za-z0-9_-]+)\.',text)
 if len(matches)!=1 or matches[0][0]!=field:raise ValueError('Unsupported authoritative update')
 return {'value':matches[0][1],'ask':None}
