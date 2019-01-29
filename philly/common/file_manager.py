from shutil import copyfile

import os


class FileManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def mkdir(self, dir):
        raise NotImplementedError

    def copyfile(self, src, dest):
        raise NotImplementedError


class APFileManager(FileManager):

    def __init__(self, base_dir):
        super().__init__(base_dir)
        import paramiko
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect("131.253.41.35", 84, "userpnrsy", "accessPnrsy")
        self.sftp = self.ssh.open_sftp()

    def mkdir(self, dir):
        self.ssh.exec_command('mkdir -p %s/%s' % (self.base_dir, dir))

    def copyfile(self, src, dest):
        self.sftp.put(src, '%s/%s' % (self.base_dir, dest))


class PhillyFileManager(FileManager):

    def mkdir(self, dir):
        os.system("philly-fs -mkdir %s/%s" % (self.base_dir, dir))

    def copyfile(self, src, dest):
        os.system("philly-fs -cp %s %s/%s" % (src, self.base_dir, dest))


class LocalFileManager(FileManager):

    def mkdir(self, dir):
        os.makedirs(self.base_dir + "/" + dir, exist_ok=True)

    def copyfile(self, src, dest):
        os.system("cp -r %s %s/%s" % (src, self.base_dir, dest))
