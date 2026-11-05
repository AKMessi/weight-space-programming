"""Bounded AST interpretation of tiny Python fixtures. No eval/exec/imports."""
import ast
import operator
import re


class Unsupported(ValueError):
    pass


BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
       ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
       ast.In: lambda a,b:a in b, ast.NotIn:lambda a,b:a not in b}
BUILTINS = {'sum':sum, 'len':len, 'min':min, 'max':max, 'abs':abs,
            'sorted':sorted, 'list':list, 'reversed':lambda x:list(reversed(x))}


class Interpreter:
    def __init__(self):
        self.steps=0

    def tick(self):
        self.steps+=1
        if self.steps>2000: raise Unsupported('step budget')

    def expression(self,node,env):
        self.tick()
        if isinstance(node,ast.Constant) and type(node.value) in (int,float,str,bool,type(None)):
            if isinstance(node.value,str) and len(node.value)>1000: raise Unsupported('string bound')
            return node.value
        if isinstance(node,ast.Name):
            if node.id not in env: raise Unsupported('unknown name')
            return env[node.id]
        if isinstance(node,(ast.List,ast.Tuple,ast.Set)):
            values=[self.expression(n,env) for n in node.elts]
            return tuple(values) if isinstance(node,ast.Tuple) else (set(values) if isinstance(node,ast.Set) else values)
        if isinstance(node,ast.Dict):
            return {self.expression(k,env):self.expression(v,env) for k,v in zip(node.keys,node.values)}
        if isinstance(node,ast.BinOp) and type(node.op) in BIN:
            a,b=self.expression(node.left,env),self.expression(node.right,env)
            if isinstance(node.op,ast.Mult) and (isinstance(a,(str,list,tuple)) or isinstance(b,(str,list,tuple))):
                if not (isinstance(a,int) and abs(a)<=100 or isinstance(b,int) and abs(b)<=100): raise Unsupported('repetition bound')
            value=BIN[type(node.op)](a,b)
            if isinstance(value,(str,list,tuple)) and len(value)>1000: raise Unsupported('result bound')
            if isinstance(value,int) and value.bit_length()>256: raise Unsupported('integer bound')
            return value
        if isinstance(node,ast.UnaryOp):
            value=self.expression(node.operand,env)
            if isinstance(node.op,ast.USub):return -value
            if isinstance(node.op,ast.UAdd):return +value
            if isinstance(node.op,ast.Not):return not value
        if isinstance(node,ast.Compare):
            a=self.expression(node.left,env)
            for op,right in zip(node.ops,node.comparators):
                b=self.expression(right,env)
                if type(op) not in CMP or not CMP[type(op)](a,b):return False
                a=b
            return True
        if isinstance(node,ast.BoolOp):
            value=None
            for item in node.values:
                value=self.expression(item,env)
                if isinstance(node.op,ast.And) and not value:return value
                if isinstance(node.op,ast.Or) and value:return value
            return value
        if isinstance(node,ast.IfExp):
            return self.expression(node.body if self.expression(node.test,env) else node.orelse,env)
        if isinstance(node,ast.Subscript):
            return self.expression(node.value,env)[self.expression(node.slice,env)]
        if isinstance(node,ast.Slice):
            return slice(*(self.expression(v,env) if v is not None else None for v in (node.lower,node.upper,node.step)))
        if isinstance(node,ast.Call) and not node.keywords:
            args=[self.expression(a,env) for a in node.args]
            if isinstance(node.func,ast.Name) and node.func.id in BUILTINS:
                return BUILTINS[node.func.id](*args)
            if isinstance(node.func,ast.Attribute) and node.func.attr in ('lower','upper','strip','split','join'):
                value=self.expression(node.func.value,env)
                if type(value) is str:return getattr(value,node.func.attr)(*args)
            raise Unsupported('call not allowed')
        if isinstance(node,(ast.ListComp,ast.GeneratorExp)) and len(node.generators)==1:
            gen=node.generators[0]
            if gen.is_async or not isinstance(gen.target,ast.Name):raise Unsupported('comprehension')
            items=list(self.expression(gen.iter,env))
            if len(items)>100:raise Unsupported('iteration bound')
            result=[]
            for item in items:
                local={**env,gen.target.id:item}
                if all(self.expression(c,local) for c in gen.ifs):result.append(self.expression(node.elt,local))
            return result
        raise Unsupported(type(node).__name__)

    def block(self,nodes,env):
        for node in nodes:
            self.tick()
            if isinstance(node,ast.Return):return True,self.expression(node.value,env) if node.value else None
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):continue
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                env[node.targets[0].id]=self.expression(node.value,env)
            elif isinstance(node,ast.AugAssign) and isinstance(node.target,ast.Name) and type(node.op) in BIN:
                env[node.target.id]=BIN[type(node.op)](env[node.target.id],self.expression(node.value,env))
            elif isinstance(node,ast.If):
                done,value=self.block(node.body if self.expression(node.test,env) else node.orelse,env)
                if done:return done,value
            elif isinstance(node,ast.For) and isinstance(node.target,ast.Name):
                items=list(self.expression(node.iter,env))
                if len(items)>100:raise Unsupported('iteration bound')
                for item in items:
                    env[node.target.id]=item
                    done,value=self.block(node.body,env)
                    if done:return done,value
                if node.orelse:raise Unsupported('for else')
            else:raise Unsupported(type(node).__name__)
        return False,None


def extract_function(text,parameters):
    # One declared extraction policy: first fenced block, otherwise whole response.
    fences=re.findall(r'```(?:python)?\s*\n(.*?)```',text,re.S)
    snippet=fences[0].strip() if fences else text.strip()
    tree=ast.parse(snippet)
    if len(list(ast.walk(tree)))>250:raise Unsupported('AST size')
    if len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef):
        fn=tree.body[0]
        if fn.decorator_list or fn.args.defaults or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg:
            raise Unsupported('signature')
        if len(fn.args.args)!=len(parameters):raise Unsupported('arity')
        return snippet,fn.body,[x.arg for x in fn.args.args],bool(fences)
    if tree.body and all(isinstance(n,(ast.Return,ast.Assign,ast.If,ast.For,ast.AugAssign)) for n in tree.body):
        return snippet,tree.body,parameters,bool(fences)
    raise Unsupported('expected one function or body')


def evaluate_code(case,text):
    expected=case['expected']
    try:
        snippet,body,parameters,fenced=extract_function(text,expected['parameters'])
        results=[]
        for test in expected['tests']:
            evaluator=Interpreter()
            done,value=evaluator.block(body,dict(zip(parameters,test['args'])))
            # JSON represents tuples as lists, but bool must never alias an integer.
            from .grading import typed_equal
            results.append(done and typed_equal(value,test['result']))
        functional=all(results)
        extra=text.replace(snippet,'').replace('```python','').replace('```','').strip()
        mode=case.get('response_mode','code')
        contract=(not fenced and text.strip()==snippet) if mode=='code' else len(re.findall(r'[A-Za-z]+',extra))>=8
        return {'success':functional and contract,'payload_correct':functional,
                'functional_correct':functional,'tests_passed':sum(results),'tests_total':len(results),
                'format_valid':contract,'note':'bounded AST semantics; explanation length is a proxy' if mode=='explain' else 'bounded AST semantics'}
    except (SyntaxError,ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,OverflowError,AttributeError) as exc:
        return {'success':False,'payload_correct':False,'functional_correct':False,
                'tests_passed':0,'tests_total':len(expected['tests']),'grading_error':str(exc)[:180]}
