"""No-network DOM/Chrome stand-ins around the actual bundled side-panel module."""
from .test_product_scope import node_check

SETUP = r"""
const assert=require('node:assert/strict');
class Element extends EventTarget {
 constructor(tag='div'){super();this.tagName=tag;this.children=[];this.value='';this.checked=false;
  this.disabled=false;this.hidden=false;this.open=false;this.dataset={};this._text='';
  this.classList={add(){},remove(){}};}
 set textContent(value){this._text=value;this.children=[];}
 get textContent(){return this._text+this.children.map(x=>x.textContent).join('');}
 get childElementCount(){return this.children.length;}
 append(...items){for(const item of items){item.parent=this;this.children.push(item);}}
 replaceChildren(...items){for(const item of this.children)item.parent=null;this.children=[];this._text='';this.append(...items);}
 remove(){if(this.parent){this.parent.children=this.parent.children.filter(x=>x!==this);this.parent=null;}}
 focus(){}
 querySelectorAll(selector){const all=this.children.flatMap(x=>[x,...x.querySelectorAll('*')]);
  if(selector==='*')return all;
  if(selector.startsWith('.'))return all.filter(x=>selector.split(',').some(s=>(x.className||'').split(' ').includes(s.trim().slice(1))));
  if(selector.startsWith('button'))return all.filter(x=>x.tagName==='button'&&(!selector.includes('[value')||['allow_once','allow_run'].includes(x.value)));
  return all.filter(x=>x.tagName==='input'&&(!selector.includes(':checked')||x.checked));}
 click(){if(this.disabled)return;this.onclick?.();this.dispatchEvent(new Event('click'));}
 showModal(){this.open=true;}
 close(value){this.returnValue=value;this.open=false;this.dispatchEvent(new Event('close'));}
}
const elements=new Map([...require('fs').readFileSync('src/sidepanel/index.html','utf8').matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Element()]));
const el=id=>elements.get(id);
global.document={getElementById:el,createElement:tag=>new Element(tag),createTextNode:text=>{const e=new Element('text');e.textContent=text;return e;}};
el('conversation').append(el('empty-state'));
el('confirmation-dialog').append(...['allow_once','allow_run'].map(value=>{const b=new Element('button');b.value=value;return b;}));

global.window=new EventTarget();window.setInterval=setInterval;
let clockNow=1000000,timerId=0;const timers=new Map();Date.now=()=>clockNow;
window.setTimeout=(fn,delay)=>{const id=++timerId;timers.set(id,{fn,at:clockNow+delay});return id;};
window.clearTimeout=id=>timers.delete(id);
function advance(ms){clockNow+=ms;for(const [id,t] of [...timers])if(t.at<=clockNow){timers.delete(id);t.fn();}}
const events={};const event=name=>({addListener:fn=>(events[name]??=[]).push(fn)});
const calls=[];let ownerGate=null;let scriptGate=null;
const owner={id:1,windowId:1,active:true,url:'http://127.0.0.1:8080/',title:'Synthetic page'};
global.chrome={runtime:{id:'a'.repeat(32),getURL:p=>'chrome-extension://'+'a'.repeat(32)+'/'+p,
 ContextType:{SIDE_PANEL:'SIDE_PANEL'},getContexts:async()=>[{contextType:'SIDE_PANEL'}]},
 storage:{local:{get:async()=>({settings:{token:'synthetic-only',backendUrl:'ws://127.0.0.1:8765',allowlist:['http://127.0.0.1:8080/*']}})},onChanged:event('settings')},
 tabs:{getCurrent:async()=>undefined,get:async()=>{calls.push('tab');if(ownerGate)await ownerGate;return owner;},
 query:async()=>[owner],onActivated:event('activated'),onUpdated:event('updated'),onRemoved:event('removed'),
 onAttached:event('attached'),onDetached:event('detached')},
 windows:{WINDOW_ID_NONE:-1,onFocusChanged:event('focus')},
 sidePanel:{getOptions:async()=>({enabled:true,path:'sidepanel/index.html?tabId=1'})},
 scripting:{executeScript:async()=>{calls.push('script');if(scriptGate)await scriptGate;throw Error('Synthetic observation boundary');}}};
global.location={href:chrome.runtime.getURL('sidepanel/index.html?tabId=1')};
const sockets=[];let autoHello=true;
class Socket extends EventTarget {
 static OPEN=1;
 constructor(){super();this.readyState=1;this.sent=[];this.closed=0;this.callbacks=[];sockets.push(this);
  queueMicrotask(()=>this.dispatchEvent(new Event('open')));}
 addEventListener(type,handler,options){super.addEventListener(type,handler,options);if(type==='message')this.callbacks.push(handler);}
 send(raw){const frame=JSON.parse(raw);this.sent.push(frame);
  if(frame.type==='hello'&&autoHello)queueMicrotask(()=>this.frame('hello_ack',{provider:'mock',...require('../shared/build-contract.json'),read_only_default:true}));}
 frame(type,payload,late=false){const e=new Event('message');e.data=JSON.stringify({id:'synthetic-frame',type,payload});
  if(late)this.callbacks.forEach(fn=>fn(e));else this.dispatchEvent(e);}
 close(){this.closed++;this.readyState=3;this.dispatchEvent(new Event('close'));}
}
global.WebSocket=Socket;
const code=buildSync({entryPoints:['src/sidepanel/sidepanel.ts'],bundle:true,write:false,
 format:'cjs',platform:'node',footer:{js:'globalThis.panelSession = browserSession; globalThis.panelTransient = () => ({prepared: preparedTitles.size, selected: !!specificPdf || !!specificPrepared, offer: !!currentFileChoices});'}}).outputFiles[0].text;
new Function('module','exports','require',code)({exports:{}},{},require);
if(data.clear_on_boot)el('clear').click();
async function drain(){for(let i=0;i<70;i++)await Promise.resolve();}
const choices={offer_id:'synthetic-offer',freshness:'state_bound',results:[{result_ref:'synthetic-current-ref',title:'Synthetic.pdf',attachment_filename:'Synthetic.pdf',content_type:'attachment',file_type:'pdf'}]};
const ready={documents:[{document_id:'a'.repeat(64),title:'Synthetic.pdf'}]};
function question(){el('input').value='Synthetic question';el('input').dispatchEvent(new Event('input'));}
function cardButton(label){return el('conversation').querySelectorAll('button').find(b=>b.textContent===label);}
function selectChoice(){cardButton('Ask about this PDF').click();}
function prepareChoice(){cardButton('Prepare locally').click();}
function selectableCards(){return el('conversation').querySelectorAll('button').filter(b=>b.textContent==='Prepare locally'&&!b.disabled).length;}
function invalidateChoices(ws){ws.frame('document_choices',{offer_id:null,results:[],state:'invalidated'});}
function refreshChoices(ws){invalidateChoices(ws);cardButton('Refresh results').click();}
function neutral(){return {messages:el('conversation').children.length,empty:!el('empty-state').hidden,
 timeline:el('timeline').children.length,status:el('run-status').textContent,tokens:el('token-usage').textContent,elapsed:el('elapsed').textContent,
 legacyAbsent:!el('documents'),cards:el('conversation').querySelectorAll('.file-card').length,
 ready:panelTransient().prepared,selected:panelTransient().selected,offer:panelTransient().offer,
 consent:el('research-remote').checked,researchMode:el('research-mode').checked,selection:el('research-selection').textContent,
 confirm:el('confirmation-dialog').open,confirmText:el('confirmation-arguments').textContent,
 input:el('input').value,stop:el('stop').disabled,pause:el('pause').disabled,resume:el('resume').disabled,
 send:el('send').disabled,owner:panelSession.ownerTabId,target:panelSession.target,mode:panelSession.mode,
 snapshots:panelSession.snapshots.size,ports:panelSession.ports.size,readOnlyRemoved:!el('read-only'),
 connection:el('status-dot').dataset.state};}
"""


def run(script, **data):
    return node_check(SETUP + '\n(async()=>{await drain();\n' + script +
        '\n})().catch(e=>{console.error(e);process.exit(1);});', data)


NEUTRAL = dict(messages=1, empty=True, timeline=0, status='Ready', tokens='0 tokens', elapsed='0.0s',
    legacyAbsent=True, cards=0, ready=0, selected=False, offer=False, consent=False, researchMode=False,
    selection='', confirm=False, confirmText='', input='',
    stop=True, pause=True, resume=True, send=False, owner=1, target=None, mode='stopped', snapshots=0,
    ports=0, readOnlyRemoved=True, connection='disconnected')
