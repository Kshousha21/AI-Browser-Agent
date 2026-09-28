"""The new native panel identities must not share an acquisition cleanup lease."""
import pytest
from .test_product_scope import node_check

SETUP = r"""
let connected;const rules=new Map();const cancellations=[];
global.chrome={runtime:{id:'a'.repeat(32),getURL:p=>'chrome-extension://'+'a'.repeat(32)+'/'+p,
 onConnect:{addListener:fn=>connected=fn}},
 tabs:{get:async id=>({id,url:'https://portal.example/portal/search'})},
 downloads:{cancel:async id=>cancellations.push(id)},
 declarativeNetRequest:{RuleActionType:{BLOCK:'block',ALLOW:'allow'},ResourceType:{MAIN_FRAME:'main_frame'},
 isRegexSupported:async()=>({isSupported:true}),
 updateSessionRules:async({addRules=[],removeRuleIds=[]})=>{for(const id of removeRuleIds)rules.delete(id);for(const rule of addRules)rules.set(rule.id,rule);}}};
load('src/background/acquisition-guard.ts');
function port(owner,extra={}){
 const messages=[],handlers=[],disconnects=[];
 const value={name:'document-acquisition-guard',sender:{id:chrome.runtime.id,
 url:chrome.runtime.getURL('sidepanel/index.html?tabId='+owner),...extra},
 onMessage:{addListener:fn=>handlers.push(fn)},onDisconnect:{addListener:fn=>disconnects.push(fn)},
 postMessage:data=>messages.push(data),send:data=>handlers.forEach(fn=>fn(data)),
 disconnect:()=>disconnects.forEach(fn=>fn())};
 connected(value);return {value,messages};
}
const url='https://portal.example/portal/download/attachments/100/shared.pdf';
async function drain(){for(let i=0;i<30;i++)await Promise.resolve();}
"""


def run(script, data=None):
    return node_check(SETUP + '\n(async()=>{\n' + script +
        '\n})().catch(e=>{console.error(e);process.exit(1);});', data or {})


@pytest.mark.parametrize('case', ['other_tab', 'foreign_extension', 'ordinary_tab', 'unbound_panel'])
def test_guard_rejects_cross_owner_or_non_panel_sender(case):
    result = run(r"""
const extra=data.case==='foreign_extension'?{id:'b'.repeat(32)}:
 data.case==='ordinary_tab'?{tab:{id:100}}:
 data.case==='unbound_panel'?{url:chrome.runtime.getURL('sidepanel/index.html')}:{};
const p=port(1,extra);p.value.send({tabId:data.case==='other_tab'?2:1,url});await drain();
console.log(JSON.stringify({rules:rules.size,ready:p.messages.some(x=>x.ready),cancellations}));
""", dict(case=case))
    assert result == dict(rules=0, ready=False, cancellations=[])


def test_distinct_panel_sessions_cannot_hold_overlapping_acquisition_leases():
    result = run(r"""
const a=port(1);a.value.send({tabId:1,url});await drain();
const b=port(2);b.value.send({tabId:2,url});await drain();
const before={a:a.messages.some(x=>x.ready),b:b.messages.some(x=>x.ready),rules:rules.size};
b.value.disconnect();await drain();const ownerIntact=rules.size===2;
a.value.disconnect();await drain();const cleared=rules.size===0;
const fresh=port(2);fresh.value.send({tabId:2,url});await drain();
const next=fresh.messages.some(x=>x.ready);fresh.value.disconnect();await drain();
console.log(JSON.stringify({before,ownerIntact,cleared,next,final:rules.size}));
""")
    assert result['before'] == dict(a=True, b=False, rules=2)
    assert result['ownerIntact'] and result['cleared'] and result['next']
    assert result['final'] == 0
