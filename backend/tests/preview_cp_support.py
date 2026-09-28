"""Actual bundled reader on synthetic cp DOM; no provider/navigation/download calls."""
import json
from .test_document_preview import IDENTITY, DOWNLOAD


def check_versioned_markup(browser, owner):
    """Structured filename/version binding; DOM metadata only, synthetic owner."""
    script = r'''async(identityJson,download)=>{
      const expected=JSON.parse(identityJson),api=globalThis.__aiBrowserAgentContent;
      const original=document.body.innerHTML,originalUrl=location.href,outcomes=[];
      const test=async(name,change,error=null)=>{
        history.replaceState({},'',originalUrl);document.body.innerHTML=original;
        const root=document.querySelector('.cp-container'),title=root.querySelector('.cp-title-container');
        const file=root.querySelector('.cp-file-title'),version=root.querySelector('.cp-version');
        const group=root.querySelector('.cp-title'),link=root.querySelector('.cp-file-controls a');
        const args={identity:{...expected},url:originalUrl,download};
        change({root,title,file,version,group,link,args});
        const result=await api.inspectAttachmentPreview(args.url,args.identity,false);
        outcomes.push({name,pass:error?result.error===error:result.preview?.url===args.download&&
          result.preview.attachment.version==='1'&&result.page_state.visible_text===''&&result.page_state.elements.length===0});
      };
      const mismatch='preview_identity_mismatch',ambiguous='preview_ambiguous';
      await test('observed filename plus current-version title',()=>{});
      await test('filename-only aria label',({root,file})=>{file.id='synthetic-filename';root.setAttribute('aria-labelledby',file.id)});
      await test('generated title IDs',({root,title})=>{title.id='cp-title-912891';root.setAttribute('aria-labelledby',title.id)});
      await test('wrong filename span',({file})=>file.textContent='other.pdf',mismatch);
      await test('wrong version',({version})=>version.textContent='v.2 [Current]',mismatch);
      await test('changed href version',({link})=>link.href=link.href.replace('version=1','version=2'),mismatch);
      await test('missing href version',({link})=>link.href=link.href.replace('version=1&',''),mismatch);
      await test('unrecognized version display',({version})=>version.textContent='latest',mismatch);
      await test('old version display unsupported',({version})=>version.textContent='v.1 [Historical]',mismatch);
      await test('version-label-only aria binding',({root,version})=>{version.id='version-only';root.setAttribute('aria-labelledby',version.id)},mismatch);
      await test('extra title text',({title})=>title.append(' another file'),mismatch);
      await test('hidden filename',({file})=>file.hidden=true,mismatch);
      await test('hidden version',({version})=>version.hidden=true,mismatch);
      await test('two filenames',({group,file})=>group.append(file.cloneNode(true)),ambiguous);
      await test('two versions',({group,version})=>group.append(version.cloneNode(true)),ambiguous);
      await test('version outside matching group',({title,version})=>title.append(version),mismatch);
      for(const target of ['file','version','group']) await test('contradictory optional '+target,
        nodes=>nodes[target].setAttribute('data-attachment-version','2'),mismatch);
      await test('encoded filename and preserved query',({file,link,args})=>{
        const name='Synthetic Pr\u00fcfung + \u5149\u5b50.pdf';file.textContent=name;args.identity.filename=name;
        const u=new URL(originalUrl);u.searchParams.set('preview','/100/200/'+name);args.url=u.href;
        history.replaceState({},'',args.url);
        link.setAttribute('href','/portal/download/attachments/100/'+encodeURIComponent(name)+'?version=1&modificationDate=123&api=v2&download=true');
        args.download=link.href;
      });
      history.replaceState({},'',originalUrl);document.body.innerHTML=original;return outcomes;
    }'''
    result = browser.ui('chrome.scripting.executeScript({target:{tabId:' + str(owner) +
        '},func:' + script + ',args:[' + json.dumps(json.dumps(IDENTITY)) + ',' + json.dumps(DOWNLOAD) +
        ']}).then(xs=>xs[0].result)')
    assert result and all(row['pass'] for row in result), [row['name'] for row in result if not row['pass']]


def check_observed_markup(browser, owner):
    # Match production's JSON-string transport: Chrome omits null object fields.
    script = r'''async(identityJson, download)=>{
      const expected=JSON.parse(identityJson);
      const api=globalThis.__aiBrowserAgentContent, original=document.body.innerHTML;
      const originalUrl=location.href, outcomes=[];
      const read=(identity=expected,url=originalUrl)=>api.inspectAttachmentPreview(url,identity,false);
      const test=async(name,change,error=null,stage=null)=>{
        document.body.innerHTML=original;
        const root=document.querySelector('.cp-container'),title=root.querySelector('.cp-title-container');
        const container=root.querySelector('.cp-file-controls'),link=container.querySelector('a');
        const args={identity:{...expected},url:originalUrl};
        change({root,title,container,link,args});
        const result=await read(args.identity,args.url);
        outcomes.push({name,pass:error?result.error===error&&(!stage||result.stage===stage):
          !!result.preview&&result.preview.url===download&&result.preview.attachment.version==='1'&&
          result.page_state.visible_text===''&&result.page_state.elements.length===0});
      };
      const missing='preview_control_missing', mismatch='preview_identity_mismatch', ambiguous='preview_ambiguous';
      await test('observed div without open or invented identity attributes',()=>{});
      await test('generated IDs',({root,title})=>{root.id='cp-123456';title.id='cp-title-987654';root.setAttribute('aria-labelledby',title.id)});
      await test('title without optional labelledby',({root})=>root.removeAttribute('aria-labelledby'));
      for(const field of ['parent_id','attachment_id','filename']) await test('selected '+field,({args})=>{
        args.identity[field]=field==='filename'?'wrong.pdf':'999';
      },mismatch,'destination');
      await test('current preview differs',({args})=>{args.url=originalUrl.replace('200','201');args.identity.attachment_id='201'},mismatch,'destination');
      await test('wrong parent',({link})=>link.href=link.href.replace('/100/','/101/'),mismatch);
      await test('wrong href filename',({link})=>link.href=link.href.replace('synthetic-0','other'),mismatch);
      await test('wrong title filename',({title})=>title.textContent='other.pdf',mismatch,'preview_title');
      await test('decorated unobserved version title',({title})=>title.textContent+=' (version 2)',mismatch,'preview_title');
      await test('missing title',({title})=>title.remove(),missing,'preview_title');
      await test('unresolved label ID',({root})=>root.setAttribute('aria-labelledby','missing'),mismatch);
      await test('label outside preview',({root})=>{const p=document.createElement('p');p.id='outside';p.textContent=expected.filename;document.body.append(p);root.setAttribute('aria-labelledby','outside')},mismatch);
      await test('duplicate title ID',({title})=>document.body.append(title.cloneNode(true)),mismatch);
      await test('two visible titles',({root,title})=>root.append(title.cloneNode(true)),ambiguous);
      await test('two visible dialogs',({root})=>document.body.append(root.cloneNode(true)),ambiguous);
      await test('hidden dialog ignored',({root})=>{const other=root.cloneNode(true);other.style.display='none';other.removeAttribute('id');other.querySelector('[id]').removeAttribute('id');document.body.append(other)});
      await test('only hidden dialog',({root})=>root.style.display='none',missing);
      await test('two controls',({container,link})=>container.append(link.cloneNode(true)),ambiguous);
      await test('hidden control ignored',({container,link})=>{const other=link.cloneNode(true);other.style.display='none';container.append(other)});
      await test('only hidden control',({link})=>link.style.display='none',missing);
      await test('disabled control',({link})=>link.setAttribute('aria-disabled','true'),missing);
      await test('disabled controls container',({container})=>container.setAttribute('aria-disabled','true'),missing);
      await test('inert control',({link})=>link.setAttribute('inert',''),missing);
      await test('wrong accessible label',({link})=>link.setAttribute('aria-label','Other'),missing);
      await test('outside preview only',({link})=>link.remove(),missing);
      await test('outside controls',({root,link})=>root.append(link),missing);
      await test('two control containers',({root,container})=>root.append(container.cloneNode(true)),ambiguous);
      await test('missing controls container',({container})=>container.className='',missing,'preview_controls');
      await test('wrong role',({root})=>root.removeAttribute('role'),missing);
      await test('nested foreign dialog',({container,link})=>{const d=document.createElement('div');d.setAttribute('role','dialog');d.append(link);container.append(d)},missing);
      for(const target of ['root','link','title']) for(const [attr,value] of [
        ['data-page-id','101'],['data-attachment-id','201'],['data-attachment-filename','wrong.pdf'],['data-attachment-version','2']]) {
        await test('contradictory '+target+' '+attr,nodes=>nodes[target].setAttribute(attr,value),mismatch);
      }
      await test('consistent optional evidence',({root,link,title})=>{
        for(const node of [root,link,title]) for(const [attr,value] of [
          ['data-page-id','100'],['data-attachment-id','200'],['data-attachment-filename',expected.filename],['data-attachment-version','1']]) node.setAttribute(attr,value);
      });
      await test('changed href version contradicts evidence',({root,link})=>{root.setAttribute('data-attachment-version','1');link.href=link.href.replace('version=1','version=2')},mismatch);
      await test('missing href version contradicts evidence',({root,link})=>{root.setAttribute('data-attachment-version','1');link.href=link.href.replace('version=1&','')},mismatch);
      await test('unsupported query',({link})=>link.href+='&unknown=1',mismatch);
      await test('duplicate query',({link})=>link.href+='&version=1',mismatch);
      await test('encoded traversal',({link})=>link.setAttribute('href','/portal/download/attachments/100/%2e%2e/wrong.pdf'),mismatch);
      document.body.innerHTML=original;
      const final=await read();
      outcomes.push({name:'restored exact target',pass:final.preview?.url===download});
      return outcomes;
    }'''
    result = browser.ui('chrome.scripting.executeScript({target:{tabId:' + str(owner) +
        '},func:' + script + ',args:[' + json.dumps(json.dumps(IDENTITY)) + ',' + json.dumps(DOWNLOAD) +
        ']}).then(xs=>xs[0].result)')
    assert result and all(row['pass'] for row in result), [row['name'] for row in result if not row['pass']]


def check_encoded_markup(browser, owner, identity, download):
    script = r'''async(identityJson,download)=>{
      const identity=JSON.parse(identityJson);
      const api=globalThis.__aiBrowserAgentContent,link=document.querySelector('.cp-file-controls a');
      const original=link.getAttribute('href'),checks=[];
      const read=()=>api.inspectAttachmentPreview(location.href,identity,false);
      const first=await read();checks.push(first.preview?.url===download&&first.preview?.attachment.version==='1');
      link.setAttribute('href',original.replace('version=1','version=2'));
      checks.push((await read()).error==='preview_identity_mismatch');
      link.setAttribute('href',original.replace('version=1&',''));
      checks.push((await read()).error==='preview_identity_mismatch');
      link.setAttribute('href',original);
      checks.push((await read()).preview?.url===download);
      return checks;
    }'''
    assert all(browser.ui('chrome.scripting.executeScript({target:{tabId:' + str(owner) +
        '},func:' + script + ',args:[' + json.dumps(json.dumps(identity)) + ',' + json.dumps(download) +
        ']}).then(xs=>xs[0].result)'))
