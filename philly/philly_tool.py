
import os
import json
import re
from copy import deepcopy
from threading import Thread
import requests
import argparse
from requests_ntlm import HttpNtlmAuth
import warnings
from philly.common.file_manager import PhillyFileManager, LocalFileManager, APFileManager
from philly import config
from multiprocessing.dummy import Pool as ThreadPool

warnings.filterwarnings('ignore')

username = config.username
login_user = config.login_user
password = config.password
project_name = config.project_name
remote_project_name = config.remote_project_name
local_ssh = config.local_ssh
best_vc = {
    'gcr': 'pnrsy',
    'rr1': 'sdrgvc',
    'rr2': 'msrlabs',
    'cam': 'msrlabs',
    'eu1': 'msrlabs',

    # 'eu2': 'pnrsy',
    'eu2': 'msrlabs',

    'sc1': 'pnrsy',
    'sc2': 'msrlabs',
    'wu1': 'msrlabs',
    'wu2': 'msrlabs',
    'et1': 'msrlabs',
}


# best_vc = {}


def kill_job(args):
    self, ii = args
    res = self.session.get('https://philly/api/abort?clusterId=%s&jobId=%s' %
                           (self.cluster_id, ii), verify=False).json()
    print(res)
    # self.response[t_id] = res


class Philly:
    def __init__(self, args):
        self.args = args
        if not isinstance(args, dict):
            args = vars(args)
        self.local = args.get('local', False)
        self.session = requests.Session()
        self.session.auth = HttpNtlmAuth('xuta', password)
        self.gpus = args.get('gpus', 1)
        self.cluster_id = args['cluster_id']
        if self.cluster_id == 'ap':
            self.cluster_id = 'philly-prod-cy4'
        self.vc_id = args.get('vc_id', None) or best_vc.get(self.cluster_id, 'sdrgvc')
        self.debug = args.get('debug', False)
        self.threads = []
        self.response = []
        self.running_job_names = []
        self.tasks = []
        self.clusters = ['eu1', 'eu2', 'sc1', 'sc2', 'rr1', 'rr2', 'gcr', 'cam', 'wu2', 'philly-prod-cy4']

    def get_job_list(self):
        my_jobs = self._get_my_job_list('all')
        runningjobs_all = [x for x in my_jobs['runningJobs']]
        queuedjobs_all = [x for x in my_jobs['queuedJobs']]

        print("========= Queue =========")
        for j in queuedjobs_all:
            self._print_job_info(j)
            print()
        print("\n========= Runnings =========")
        for j in runningjobs_all:
            self._print_job_info(j)
            print()

    def get_debug_job_ip(self, return_export_str=True):
        
        # debug_job = [x for x in self._get_my_job_list('all')['runningJobs']][-1]
        debug_job = [x for x in self._get_my_job_list('all', username=username)['runningJobs']][-1]
        #print(debug_job)
        debug_job_ip = debug_job['detail'][1]['ip']
        debug_job_port = debug_job['detail'][1]['port']
        if int(debug_job_port) == 0:
            debug_job_ip = debug_job['detail'][0]['ip']
            debug_job_port = debug_job['detail'][0]['port']
        #print(debug_job_ip, debug_job_port)       
 
        debug_job_ip = "10.228.130.46"
        debug_job_port="2280" 
        if return_export_str:
            return "export ip={} port={}".format(debug_job_ip, debug_job_port)
        else:
            return debug_job_ip, debug_job_port

    def kill_jobs(self, task_name=""):
        jobs = self._get_my_job_list('all')
        jobs = jobs['queuedJobs'] + jobs['runningJobs']
        jobid = [x["appID"] for x in jobs if task_name in x['name']]
        jobname = [x["name"] for x in jobs if task_name in x['name']]
        print(jobname)
        conf = input('please comfirm the jobid, type yes to kill them\n')
        pool = ThreadPool(10)
        if conf == 'yes':
            pool.map(kill_job, [(self, ii) for ii in jobid])

    def prepare_submit(self):
        if self.local:
            return
        #cmd = "export vc=%s ; bash philly/sync_local2philly.sh %s" % \
        #      (self.vc_id, self.cluster_id)
        #os.system(cmd)
        jobs = self._get_my_job_list('all')
        jobs = jobs['queuedJobs'] + jobs['runningJobs']
        self.running_job_names = set([x['name'].split("-")[2].split("!")[0] for x in jobs])

    def add_task(self, script, exp_name='default', save_ckpt=False, gpus=None, distributed=False):
        self.tasks.append((script, exp_name, save_ckpt, gpus, distributed))

    def create_debug_container(self):
        self.submit_job(('config_debug', 2, False))

    def get_all_logs(self):

        def print_log(job, lines, c):
            print("============  BEGIN ============")
            self._print_job_info(job, c)
            print("--------------------------------------------")
            print(self._get_log(job['appID'], lines, c))
            print("\n--------------------------------------------")
            self._print_job_info(job, c)
            print("============   END  ============\n")

        def get_logs_one_cluster(c):
            all_jobs = self._get_my_job_list('all', c)
            jobs = all_jobs['runningJobs']
            if args.scope == 'all':
                jobs += all_jobs['finishedJobs']
            jobs = [j for j in jobs if 'ns-p-debug' not in j['name'] and args.task_name in j['name']]
            if len(jobs) > 0:
                print("\n>>>>>>>>>>>>>>  %s  <<<<<<<<<<<<<\n" % c)
                if self.args.index == -1:
                    for job in jobs:
                        print_log(job, args.lines, c)
                else:
                    job = jobs[self.args.index]
                    print_log(job, args.lines * 5, c)

        if self.cluster_id == 'all':
            for c in self.clusters:
                get_logs_one_cluster(c)
        else:
            get_logs_one_cluster(self.cluster_id)

    def start(self):
        jobs = []
        for t in self.tasks:
            script, exp_name, save_ckpt, gpus, distributed = t
            if gpus > 4:
                distributed = True
            if self.local:
                try:
                    os.system(script + " --exp-name={}".format(exp_name))
                except Exception:
                    print("Run error. Exit.")
                    break
            else:
                job_name = self.create_job(script, exp_name, save_ckpt, distributed)
                if job_name is not None:
                    jobs.append((job_name, gpus, distributed))
                    # threads.append(Thread(target=self.submit_job, args=(job_name, gpus, distributed), daemon=True))
        pool = ThreadPool(10)
        pool.map(self.submit_job, jobs)

    def create_job(self, script, exp_name, save_ckpt, distributed):
        try:
            # script2jobname = script.replace('.sh', '')
            # script2jobname = re.sub(" ", "", script2jobname)
            # script2jobname = re.sub("--.*?ckpt.*?--", "", script2jobname)
            # script2jobname = re.sub("[.,/\-]+", "_", script2jobname)
            job_name = re.sub("[.,/\-]", "_", str(exp_name))
        except Exception as e:
            print(e)
        job_name = job_name.lower()
        if job_name in self.running_job_names and not self.local:
            return None

        if distributed:
            script = script + " --distributed=True "
        if exp_name != '':
            script = script + " --exp-name={} ".format(exp_name)

        os.makedirs('submitted_scripts', exist_ok=True)
        script_local = r"submitted_scripts/%s.sh" % job_name
        with open(script_local, "w", newline="\n") as f:
            lines = [
                '#!/bin/bash',
                'nvidia-smi',
                'exit_status=0',
                'cd /var/storage/shared/%s/%s/%s' % (self.vc_id, username, remote_project_name),
                'source philly/configure_philly.sh %s' % self.vc_id,
                '%s' % script,
                'wait', 
                'exit $exit_status',
            ]
            f.writelines([l.strip() + "\n" for l in lines])
        return job_name

    def submit_job(self, args):
        job_name, gpus, distributed = args
        script_local = r"submitted_scripts/%s.sh" % job_name

        if job_name != 'config_debug':
            config_file = "/var/storage/shared/%s/%s/scripts/%s.sh" % (self.vc_id, username, job_name)
            if self.is_azure():
                mpi_args = "NCCL_SOCKET_IFNAME=eth0 "
                base_dir = "gfs://%s/%s/%s" % (self.cluster_id, self.vc_id, username)
                file_manager = PhillyFileManager(base_dir)
                file_manager.mkdir('scripts')
                file_manager.copyfile(script_local, "scripts/%s.sh" % job_name)
            else:
                # if self.is_ap():
                #     mpi_args = 'NCCL_IB_HCA=mlx5_0,mlx5_2 NCCL_SOCKET_IFNAME=net0'
                # else:
                # mpi_args = 'NCCL_IB_HCA="mlx5_0,mlx5_1" NCCL_SOCKET_IFNAME=ib1'
                # mpi_args = 'NCCL_IB_HCA="mlx5_0,mlx5_1" NCCL_SOCKET_IFNAME=ib0'
                # mpi_args = "env NCCL_SOCKET_IFNAME=ib1 "
                ip, port = self.get_debug_job_ip(return_export_str=False)
                cmd = "scp -r -P {} {} {}@{}:/var/storage/shared/{}/{}/scripts/".format(port, script_local, username, ip,
                                                                                    self.vc_id, username)
                print(cmd) 
                os.system(cmd)

            submit_json = {
                'UserName': login_user,
                'Inputs': [{
                    'Path': '/hdfs/%s' % self.vc_id,
                    'Name': 'dataDir',
                }],
                'IsCrossRack': False,
                "IsMemCheck": False,
                "RackId": "anyConnected",
                "MinGPUs": self.gpus if gpus is None else gpus,
                "ToolType": None,
                "BuildId": 0,
                "Outputs": [],
                "ClusterId": self.cluster_id,
                "IsDebug": self.debug,
                "JobName": job_name,
                "ConfigFile": config_file,
                "Tag": "tf14-py35",
                "Repository": "philly/jobs/custom/tensorflow",
                "PrevModelPath": None,
                "Registry": "phillyregistry.azurecr.io",
                "VcId": self.vc_id,
                "SubmitCode": "p",
            }
                #"Tag": "pytorch-0.4.0-gloo-py36",
            submit_json['OneProcessPerContainer'] = True
            if not distributed:
                submit_json['NumOfContainers'] = '1'
                submit_json['dynamicContainerSize'] = False
            else:
                submit_json['dynamicContainerSize'] = True
                # submit_json['customMPIArgs'] = 'env NCCL_DEBUG=INFO %s' % mpi_args
        else:
            config_file = "/var/storage/shared/%s/%s/%s.sh" % (self.vc_id, username, job_name)
            submit_json = {'UserName': login_user, 'Inputs': [{
                'Path': '/hdfs/%s' % self.vc_id,
                'Name': 'dataDir',
            }], 'IsCrossRack': False, "IsMemCheck": False, "RackId": "anyConnected",
                           "MinGPUs": self.gpus if gpus is None else gpus, "ToolType": None,
                           "Tag": "pytorch-0.4.0-gloo-py36", "BuildId": 0, "Outputs": [], "ClusterId": self.cluster_id,
                           "IsDebug": self.debug, "JobName": job_name, "ConfigFile": config_file,
                           "Repository": "philly/jobs/custom/pytorch", "PrevModelPath": None,
                           "Registry": "phillyregistry.azurecr.io", "VcId": self.vc_id, "SubmitCode": "p",
                           'NumOfContainers': '1', 'dynamicContainerSize': False, 'OneProcessPerContainer': True}

        json_fn = 'submitted_scripts/%s.json' % job_name
        with open(json_fn, 'w') as f:
            json.dump(submit_json, f)

        with open(json_fn) as f:
            res = self.session.post(url='https://philly/api/v2/submit',
                                    json=json.load(f), verify=False)
            try:
                print(res.json(), job_name)
            except Exception as e:
                print("[%s] failed." % job_name, e)

    def is_azure(self):
        if 'eu' in self.cluster_id or 'sc' in self.cluster_id or 'wu' in self.cluster_id or 'et' in self.cluster_id:
            return True
        return False

    def is_ap(self):
        if 'philly-prod-cy4' in self.cluster_id:
            return True
        return False

    def _get_my_job_list(self, status, c_id=None, username=None):
        res = self.session.get(
            'https://philly/api/list?clusterId=%s&numFinishedJobs=500&vcId=%s&userName=%s&status=%s'
            % (c_id or self.cluster_id, self.vc_id, username or login_user, status), verify=False).json()
        return res

    def _get_job_list(self, status):
        return self.session.get(
            'https://philly/api/list?clusterId=%s&vcId=%s&status=%s'
            % (self.cluster_id, self.vc_id, status), verify=False).json()

    def _get_log(self, job_id, lines, cluster_id=None):
        cluster_id = cluster_id or self.cluster_id
        text = self.session.get(
            'https://philly/api/log?clusterId=%s&vcId=%s&jobId=%s&logType=stdout&logRev=latest&content=full&jobType=ns'
            % (cluster_id, self.vc_id, job_id), verify=False).text.strip()
        text = "\n".join(text.split("\n")[-lines:])
        return text

    def _print_job_info(self, job, cluster_id=None):
        cluster_id = cluster_id or self.cluster_id
        job_id = job['appID'][12:]
        urls = [
            "[Logs]: https://philly/#/logs/%s/%s/%s/%s/stdout/latest/ns" % (
                cluster_id, self.vc_id, login_user, job_id),
            "[Monitor]: https://philly/#/job/%s/%s/%s" % (
                cluster_id, self.vc_id, job['appID'][12:]),
            "[Cluster]: https://philly/#/jobSummary/%s/all/%s" % (
                cluster_id, login_user),
            "!!!!!!!!!!!!!![KILL]: https://philly/api/abort?clusterId=%s&jobId=%s" % (
                cluster_id, job['appID']),
        ]
        urls = "\n".join(urls)

        print("cluster: %s\nJob name: %s\nStarted at: %s\nGPUS: %s" % (
            cluster_id, self._get_simple_name(job['name']), job['startDateTimes'], job['name'].split('!')[-1]))
        print(urls)

    def _get_simple_name(self, s):
        return s.split("-")[2].split("!")[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('action', help='action')
    parser.add_argument('-c', '--cluster-id', default='rr1', help='which cluster')
    parser.add_argument('-v', '--vc-id', default=None, help='which vc')
    parser.add_argument('-t', '--task_name', default='', type=str)

    # list
    action_list = parser.add_argument_group('List action')
    action_list.add_argument('-s', '--scope', default='running')

    # submit
    action_submit = parser.add_argument_group('Submit')
    action_submit.add_argument('--script', default='train.sh')
    action_submit.add_argument('--gpus', default=1, type=int)
    action_submit.add_argument('--debug', default=True, type=bool)

    # kill all
    action_kill_all = parser.add_argument_group('kill_all action')

    # log
    action_log = parser.add_argument_group('log action')
    action_log.add_argument('-l', '--lines', default=40, type=int)
    action_log.add_argument('-i', '--index', default=-1, type=int)

    args = parser.parse_args()
    os.linesep = '\n'

    philly = Philly(args)

    if args.action == 'list':
        philly.get_job_list()
    elif args.action == 'kill':
        philly.kill_jobs(args.task_name)
    elif args.action == 'debug_ip':
        print(philly.get_debug_job_ip())
    elif args.action == 'create_debug':
        philly.create_debug_container()
    elif args.action == 'log':
        philly.get_all_logs()
