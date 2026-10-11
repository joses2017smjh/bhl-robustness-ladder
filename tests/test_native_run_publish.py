"""Publication guards and actual concurrent Git fastforward behavior; no network."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("native_publisher_test",ROOT/"scripts/bench/native_run_publish.py")
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)
AUDIT=MODULE.load("native_retained_payload_test",ROOT/"scripts/bench/native_run_audit.py")


def retained_fixture(root):
    """Small original frozen intake plus a separately retained source archive."""
    original={"source/example.py":b"ORIGINAL SOURCE\n","inputs/sensor.bin":b"ORIGINAL INPUT\n"}
    manifest={name:{"bytes":len(value),"sha256":hashlib.sha256(value).hexdigest()} for name,value in original.items()}
    MODULE.write(root/"manifest.json",manifest);MODULE.write(root/"protocol.json",{"scope":"fixture"})
    retained=root/"retained.tar.gz"
    with tarfile.open(retained,"w:gz") as stream:
        for name,value in {"source/example.py":original["source/example.py"],"unused-receipt.json":b"{}"}.items():
            member=tarfile.TarInfo(name);member.size=len(value);stream.addfile(member,io.BytesIO(value))
    sensor=root/"sensor.bin";sensor.write_bytes(original["inputs/sensor.bin"])
    members={}
    for name,value in original.items():
        members[name]={"bytes":len(value),"sha256":hashlib.sha256(value).hexdigest(),"reconstruction":
            {"kind":"retained_source_archive_member","path":str(retained),"member":name} if name.startswith("source/")
            else {"kind":"surviving_exact_file","path":str(sensor)}}
    for name in ("manifest.json","protocol.json"):
        path=root/name;members[name]={"bytes":path.stat().st_size,"sha256":MODULE.digest(path),"reconstruction":{"kind":"surviving_exact_file","path":str(path)}}
    intake={"archive_sha256":"1"*64,"source_and_input_manifest_sha256":MODULE.digest(root/"manifest.json")}
    inventory={"original_archive_sha256":intake["archive_sha256"],"original_manifest_sha256":intake["source_and_input_manifest_sha256"],
        "retained_source_archive_sha256":MODULE.digest(retained),"retained_source_archive_bytes":retained.stat().st_size,"members":members}
    MODULE.write(root/"retained-member-inventory.json",inventory)
    receipt={"removed_archive_sha256":intake["archive_sha256"],"inventory_sha256":MODULE.digest(root/"retained-member-inventory.json"),
        "retained_source_archive_sha256":inventory["retained_source_archive_sha256"]}
    MODULE.write(root/"successful-packaging-reclamation.json",receipt)
    return intake,inventory,receipt


class NativePublisherTests(unittest.TestCase):
    def test_retained_subset_checks_original_intake_without_claiming_gzip_rehash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);intake,_,_=retained_fixture(root)
            actual=AUDIT.verify_retained_payloads(root,intake)
            self.assertFalse(actual["original_archive_rehashed"])
            self.assertEqual(actual["logical_frozen_payloads_verified"],4)

    def test_self_consistent_retirement_manifest_cannot_change_original_intake_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);intake,inventory,receipt=retained_fixture(root)
            manifest=json.loads((root/"manifest.json").read_text());manifest["source/example.py"]["sha256"]="2"*64
            MODULE.write(root/"manifest.json",manifest)
            inventory["original_manifest_sha256"]=MODULE.digest(root/"manifest.json")
            inventory["members"]["manifest.json"].update(bytes=(root/"manifest.json").stat().st_size,sha256=inventory["original_manifest_sha256"])
            inventory["members"]["source/example.py"]["sha256"]="2"*64
            MODULE.write(root/"retained-member-inventory.json",inventory)
            receipt["inventory_sha256"]=MODULE.digest(root/"retained-member-inventory.json")
            MODULE.write(root/"successful-packaging-reclamation.json",receipt)
            with self.assertRaisesRegex(ValueError,"immutable intake"):
                AUDIT.verify_retained_payloads(root,intake)

    def test_changed_retained_source_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);intake,_,_=retained_fixture(root)
            with (root/"retained.tar.gz").open("ab") as stream:stream.write(b"changed")
            with self.assertRaisesRegex(ValueError,"archive hash"):
                AUDIT.verify_retained_payloads(root,intake)

    def test_missing_original_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);intake,_,_=retained_fixture(root);(root/"sensor.bin").unlink()
            with self.assertRaises(FileNotFoundError):AUDIT.verify_retained_payloads(root,intake)

    def test_frozen_publisher_rejects_later_source_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);plan=root/"plan.json";bundle=root/"bundle"
            MODULE.write(plan,{"schema":"bhl-native-publish-plan-v1","repository":MODULE.REPOSITORY,
                "release_tag":MODULE.RELEASE_TAG,"collection_id":"fixture-only","commit_name":"Test",
                "commit_email":"test@example.invalid","campaigns":[{"label":"terrain","kind":"terrain",
                    "job_id":"99","job_name":"run","campaign_dir":str(root),"source_archive_sha256":"1"*64}]})
            receipt=MODULE.freeze(ROOT,plan,bundle)
            MODULE.verify_bundle(bundle,receipt["manifest_sha256"])
            changed=bundle/"source/scripts/bench/native_run_audit.py"
            changed.write_text(changed.read_text()+"\n# source changed after freeze\n")
            with self.assertRaisesRegex(ValueError,"source hash changed"):
                MODULE.verify_bundle(bundle,receipt["manifest_sha256"])

    def test_missing_completion_has_unknown_episodes_and_no_raw_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);campaign=root/"campaign";campaign.mkdir();out=root/"collected";out.mkdir()
            (campaign/"protocol.json").write_text('{}\n');pin="1"*64
            MODULE.write(campaign/"intake.json",{"archive_sha256":pin,"protocol_sha256":MODULE.digest(campaign/"protocol.json")})
            MODULE.write(campaign/"run-submission.json",{"status":"SUBMITTED","job_id":"99","archive_sha256":pin})
            observed=MODULE.collect_one({"label":"interrupted","kind":"terrain","job_id":"99","job_name":"run",
                "campaign_dir":str(campaign),"source_archive_sha256":pin},out)
            self.assertEqual(observed["scientific_status"],"INCOMPLETE")
            self.assertIsNone(observed["measured_episodes"])
            self.assertNotIn("raw",observed)

    def test_native_binary_disguised_as_raw_lidar_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive=Path(temporary)/"raw.tar.gz"
            with tarfile.open(archive,"w:gz") as stream:
                payload=b'\x7fELFfake executable';member=tarfile.TarInfo("native/input.bin");member.size=len(payload)
                stream.addfile(member,io.BytesIO(payload))
            with self.assertRaisesRegex(ValueError,"native binary"):
                MODULE.guard_raw_archive(archive)

    def test_release_upload_uses_unique_actual_filename_and_server_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            work=Path(temporary);payload=work/"outputs.tar.gz";payload.write_bytes(b"TEST RAW FIXTURE")
            checksum=MODULE.digest(payload);name="terrain-99-raw.tar.gz";commands=[]
            def fake_run(argv,**kwargs):
                commands.append(list(map(str,argv)))
                if "upload" in argv:
                    self.assertEqual(Path(argv[4]).name,name)
                    self.assertTrue(Path(argv[4]).exists())
                    return ""
                assets=[] if len(commands)==1 else [{"name":name,"size":payload.stat().st_size,"digest":"sha256:"+checksum,"id":7,"browser_download_url":"https://example.invalid/raw"}]
                return json.dumps({"assets":assets})
            with patch.object(MODULE,"run",side_effect=fake_run):
                observed=MODULE.release_asset("/fake/gh",payload,name,checksum,work)
            self.assertEqual(observed["verification"],"github_server_sha256")
            self.assertEqual(observed["sha256"],checksum)

    def test_clean_retry_preserves_concurrent_remote_commit_without_force(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);remote=root/"remote.git";seed=root/"seed";work=root/"work";work.mkdir()
            def git(*args,cwd=None):
                return subprocess.check_output(["/bin/git",*map(str,args)],cwd=cwd,stderr=subprocess.DEVNULL,text=True)
            git("init","--bare",remote);git("init","-b","main",seed)
            (seed/"seed.txt").write_text("initial\n")
            git("add",".",cwd=seed);git("-c","user.name=Test","-c","user.email=test@example.invalid","commit","-m","initial",cwd=seed)
            git("remote","add","origin",remote,cwd=seed);git("push","origin","main",cwd=seed)
            collected=root/"collected";leaf=collected/"terrain";leaf.mkdir(parents=True)
            record={"label":"terrain","job_id":"99","source_archive_sha256":"1"*64,
                    "scientific_status":"NEGATIVE","audit_status":"PASS","measured_episodes":18}
            MODULE.write(leaf/"publication-observation.json",record)
            original=MODULE.run;injected=False;clones=0;commands=[]
            def intercept(argv,**kwargs):
                nonlocal injected,clones
                argv=list(argv);commands.append(list(map(str,argv)))
                if len(argv)>1 and argv[1]=="clone":argv[-2]=str(remote);clones+=1
                result=original(argv,**kwargs)
                if "commit" in argv and not injected:
                    injected=True
                    (seed/"concurrent.txt").write_text("preserve remote work\n")
                    git("add",".",cwd=seed);git("-c","user.name=Test","-c","user.email=test@example.invalid","commit","-m","concurrent",cwd=seed);git("push","origin","main",cwd=seed)
                return result
            plan={"gh":"/fake/gh","git":"/bin/git","collection_id":"test-only","commit_name":"Test","commit_email":"test@example.invalid"}
            with patch.object(MODULE,"run",side_effect=intercept):observed=MODULE.push_records(plan,[record],collected,work)
            self.assertEqual(observed["status"],"PUSHED");self.assertEqual(clones,2)
            verify=root/"verify";git("clone","--branch","main",remote,verify)
            self.assertEqual((verify/"concurrent.txt").read_text(),"preserve remote work\n")
            self.assertTrue((verify/MODULE.RESULTS/"collection-index.json").is_file())
            self.assertFalse(any("--force" in command or "+HEAD:main" in command for command in commands))


if __name__=="__main__":unittest.main()
