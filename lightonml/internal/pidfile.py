import atexit
import errno
import os


class PidFile:
    def __init__(self, path, log=None, warn=None):
        self.pidfile = path
        self.log = lambda x: print(x, file=log) if log else blank_fn
        self.warn = lambda x: print(x, file=warn) if warn else blank_fn

    def __enter__(self):
        # Open flags: create write-only file, and Ensure that this call creates the file
        flags = os.O_CREAT | os.O_WRONLY | os.O_EXCL
        self.pidfd = None
        try:
            dir_ = os.path.dirname(self.pidfile)
            if not os.path.isdir(dir_):
                os.makedirs(dir_, mode=0o777, exist_ok=True)
            if os.stat(dir_).st_mode & 0o777 != 0o777:
                # directory isn't writeable by all, try to chmod it
                try:
                    os.chmod(dir_, 0o777)
                except OSError as exc:
                    # Failed chmod is not an error, try to continue
                    self.warn("Changing permission on directory failed: " + str(exc))
            self.pidfd = os.open(self.pidfile, flags)
            # chmod so that everyone can take over the pid if the process has crashed
            # Note that using mode argument of os.open doesn't work because of user's umask
            os.fchmod(self.pidfd, 0o777)
            os.write(self.pidfd, bytes(str(os.getpid()), 'ascii'))
        except FileExistsError:
            pid = self._check()
            if pid:
                # Process is still running
                self.pidfd = None
                raise ProcessRunningException("process already running in {} as pid {}\n"
                                              .format(self.pidfile, pid))
            else:
                # Process is dead, try to remove the PID
                os.remove(self.pidfile)
                self.warn("removed staled lockfile {}".format(self.pidfile))
                # And re-enter to try again
                return self.__enter__()
        finally:
            if self.pidfd:
                try:
                    os.close(self.pidfd)
                except OSError as e:
                    self.log("Error at close: " + str(e))
        self._register_atexit()
        self.log("locked pidfile {}".format(self.pidfile))

        return self

    def __exit__(self, t, e, tb):
        # return false to raise, true to pass
        if t is None:
            # normal condition, no exception
            self._remove()
            # atexit registry not needed anymore
            atexit.unregister(self._delete_func)
            return True
        elif t is ProcessRunningException:
            # do not remove the other process lockfile
            return False
        else:
            # other exception
            if self.pidfd:
                # this was our lockfile, removing
                self._remove()
                atexit.unregister(self._delete_func)
            return False

    def _remove(self):
        os.remove(self.pidfile)
        self.log("removed pidfile {}".format(self.pidfile))

    def _register_atexit(self):
        """Register deletion function at process exit

        This in case the context manager isn't exited properly"""

        # Generate unique name for atexit deletion, since atexit makes no distinction
        # if multiple functions are registered with the same name
        # We can use the base name of file, which is uniquely related to the object
        func_name = "pidfile_delete_" + os.path.basename(self.pidfile)
        # remove invalid identifiers
        func_name = func_name.translate(str.maketrans(" .-", "___"))
        # generate function
        delete_func = make_delete_func(func_name)
        # record it for calling unregister at exit
        self._delete_func = delete_func
        atexit.register(delete_func, self)

    def _check(self) -> int or False:
        """
        Checks if the process in PID file is still running

        The process id is expected to be in pidfile, which should exist.
        if it is still running, returns the pid, if not, return False.
        """
        with open(self.pidfile, 'r') as f:
            try:
                pidstr = f.read()
                pid = int(pidstr)
            except ValueError:
                # not an integer
                self.log("not an integer: {}".format(pidstr))
                return False
            if pid_exists(pid):
                return pid
            else:
                return False


class ProcessRunningException(BaseException):
    pass


def pid_exists(pid: int):
    """Check whether pid exists in the current process table.
    UNIX only.
    """
    if pid < 0:
        return False
    if pid == 0:
        # According to "man 2 kill" PID 0 refers to every process
        # in the process group of the calling process.
        # On certain systems 0 is a valid PID but we have no way
        # to know that in a portable fashion.
        raise ValueError('invalid PID 0')
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            # ESRCH == No such process
            return False
        elif err.errno == errno.EPERM:
            # EPERM clearly means there's a process to deny access to
            return True
        else:
            # According to "man 2 kill" possible error values are
            # (EINVAL, EPERM, ESRCH)
            raise
    else:
        return True


def blank_fn(*_, **__):
    pass


def make_delete_func(name: str):
    """Returns a PidFile remove function of specified name,
    in order to have unique atexit registration"""
    # noinspection PyProtectedMember,PyBroadException
    def delete_func(pidfile_):
        # noinspection PyPep8
        try:
            pidfile_._remove()
        except:
            pass
    assert name.isidentifier()
    delete_func.__name__ = name
    return delete_func
