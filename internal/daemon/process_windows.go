//go:build windows

package daemon

import (
	"fmt"
	"os"
	"syscall"
)

const windowsSynchronize = 0x00100000

// IsProcessRunning checks if a process with the given PID is running.
func IsProcessRunning(pid int) bool {
	handle, err := syscall.OpenProcess(windowsSynchronize, false, uint32(pid))
	if err != nil {
		return false
	}
	defer func() { _ = syscall.CloseHandle(handle) }()

	event, err := syscall.WaitForSingleObject(handle, 0)
	return err == nil && event == syscall.WAIT_TIMEOUT
}

// StopProcess terminates a process on Windows.
func StopProcess(pid int) error {
	process, err := os.FindProcess(pid)
	if err != nil {
		return fmt.Errorf("cannot find process: %w", err)
	}

	if err := process.Kill(); err != nil {
		return fmt.Errorf("cannot terminate process: %w", err)
	}

	_, _ = process.Wait()
	return nil
}
