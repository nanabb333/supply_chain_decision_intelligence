import importlib.util,json,os,hashlib
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('safe_publication',Path(__file__).parents[1]/'tools/safe_publication.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
@pytest.fixture
def env(tmp_path):
 root=tmp_path/'protected';root.mkdir();(root/'evidence').write_text('accepted');return root,{'a.txt':b'accepted','b.txt':b'other'}
@pytest.mark.parametrize('args',[[],['build'],['verify']])
def test_missing(args):
 with pytest.raises(SystemExit):m.main(args)
@pytest.mark.parametrize('which',['root','child','ancestor','traversal'])
def test_protected(tmp_path,env,which):
 root,files=env;p={'root':root,'child':root/'new','ancestor':tmp_path,'traversal':tmp_path/'new'/'..'/'out'}[which]
 with pytest.raises(m.UnsafeDestination):m.build(str(p),files,root)
 assert (root/'evidence').read_text()=='accepted'
@pytest.mark.parametrize('kind',['empty','occupied','file'])
def test_collision(tmp_path,env,kind):
 root,files=env;p=tmp_path/'out'
 if kind=='file':p.write_text('sentinel')
 else:
  p.mkdir()
  if kind=='occupied':(p/'sentinel').write_text('untouched')
 with pytest.raises(OSError):m.build(str(p),files,root)
 assert not (p/'OWNER.json').exists()
@pytest.mark.parametrize('where',['parent','target'])
def test_symlink(tmp_path,env,where):
 root,files=env;link=tmp_path/'link';link.symlink_to(root,target_is_directory=True)
 with pytest.raises(m.UnsafeDestination):m.build(str(link/'out' if where=='parent' else link),files,root)
 assert (root/'evidence').read_text()=='accepted'
def test_partial_failure_no_cleanup(tmp_path,env,monkeypatch):
 root,files=env;p=tmp_path/'out';sentinel=tmp_path/'unrelated';sentinel.write_text('keep');original=m.write_new
 def fail(fd,name,data):
  if name=='b.txt':raise OSError('injected failure')
  original(fd,name,data)
 monkeypatch.setattr(m,'write_new',fail)
 with pytest.raises(OSError):m.build(str(p),files,root)
 assert (p/'a.txt').read_bytes()==b'accepted' and not (p/'MANIFEST.json').exists()
 assert sentinel.read_text()=='keep' and (root/'evidence').read_text()=='accepted'
 with pytest.raises(ValueError):m.verify(str(p),files,root)
def test_readonly_idempotent(tmp_path,env,monkeypatch):
 root,files=env;p=tmp_path/'out';m.build(str(p),files,root)
 snapshot=lambda:{x.name:(x.read_bytes(),x.stat().st_mtime_ns) for x in p.iterdir()}
 before=snapshot();monkeypatch.setattr(m,'write_new',lambda *a:pytest.fail('write during verification'))
 assert m.verify(str(p),files,root)==m.verify(str(p),files,root);assert snapshot()==before
@pytest.mark.parametrize('change',['extra','content','symlink','manifest'])
def test_tamper(tmp_path,env,change):
 root,files=env;p=tmp_path/'out';m.build(str(p),files,root)
 if change=='extra':(p/'extra').write_text('x')
 elif change=='content':(p/'a.txt').write_text('bad')
 elif change=='manifest':(p/'MANIFEST.json').write_text('{}')
 else:(p/'a.txt').unlink();(p/'a.txt').symlink_to(root/'evidence')
 with pytest.raises((ValueError,OSError)):m.verify(str(p),files,root)
def test_unsafe_temp_name(tmp_path,env):
 root,_=env
 with pytest.raises(ValueError):m.build(str(tmp_path/'out'),{'../escape':b'x'},root)
 assert not (tmp_path/'escape').exists()
def test_external_symlink_parent(tmp_path,env):
 root,files=env;real=tmp_path/'real';real.mkdir();link=tmp_path/'link';link.symlink_to(real,target_is_directory=True)
 with pytest.raises(m.UnsafeDestination):m.build(str(link/'out'),files,root)
 assert not list(real.iterdir())
def test_corrupt_source_before_write(tmp_path,env,monkeypatch):
 root,_=env;source=root/m.SOURCE;source.mkdir(parents=True);(source/'a.txt').write_text('bad');base=tmp_path/'baseline.json';base.write_text(json.dumps({m.SOURCE+'a.txt':hashlib.sha256(b'good').hexdigest()}));monkeypatch.setattr(m,'ASSETS',['a.txt'])
 with pytest.raises(ValueError):m.payload(root,base)
 assert (source/'a.txt').read_text()=='bad'
def test_failure_cleanup_owned_temp(tmp_path,env):
 import tempfile
 root,files=env;sentinel=tmp_path/'unrelated';sentinel.write_text('keep')
 with tempfile.TemporaryDirectory(dir=tmp_path) as owned:
  out=Path(owned)/'out';m.build(str(out),files,root)
  assert m.verify(str(out),files,root)['status']=='PASS'
 assert not Path(owned).exists() and sentinel.read_text()=='keep' and (root/'evidence').read_text()=='accepted'
