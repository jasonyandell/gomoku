# Autolab launchd plists (full-fidelity archive)

> **Archived 2026-07-04.** The literal `~/Library/LaunchAgents/com.gomoku.autolab.*.plist`
> XML + the arena/monitor/research plist descriptions, lifted verbatim from
> [autolab-supervisor-and-monitor.md](../../topics/autolab-supervisor-and-monitor.md) §(b).
> The autolab is **DORMANT** (autonomous derby stopped — see [derby.md](../../derby.md)), so
> these literal configs are archival. No facts were deleted; restore from here if the lab is revived.

---

### Literal plist — `~/Library/LaunchAgents/com.gomoku.autolab.train.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.gomoku.autolab.train</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/uv</string>
    <string>run</string><string>python</string>
    <string>-m</string><string>gomoku.lab.trainer</string>
    <string>--prod</string>
    <string>--stop-file</string><string>/Users/jason/data/autolab/stop</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>WorkingDirectory</key><string>/Users/jason/code/gomoku</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>AUTOLAB_HOME</key><string>/Users/jason/data/autolab</string>
    <key>HOME</key><string>/Users/jason</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>PYTORCH_ENABLE_MPS_FALLBACK</key><string>1</string>
    <key>HF_HUB_DISABLE_PROGRESS_BARS</key><string>1</string>
    <key>WANDB_MODE</key><string>offline</string>
  </dict>
  <key>StandardOutPath</key><string>/Users/jason/data/autolab/logs/train.out.log</string>
  <key>StandardErrorPath</key><string>/Users/jason/data/autolab/logs/train.err.log</string>
  <key>ProcessType</key><string>Standard</string>
  <key>Nice</key><integer>5</integer>
</dict>
</plist>
```

`com.gomoku.autolab.arena.plist` — **identical** except `Label` =
`com.gomoku.autolab.arena`, `ProgramArguments` =
`[/opt/homebrew/bin/uv, run, python, -m, gomoku.lab.arena, --stop-file, /Users/jason/data/autolab/stop]`,
and `Standard{Out,Err}Path` → `arena.{out,err}.log`. (No `--prod`; the arena has
no MVP/prod cap.)

`com.gomoku.autolab.monitor.plist` — periodic, **no KeepAlive**:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.gomoku.autolab.monitor</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/uv</string>
    <string>run</string><string>python</string>
    <string>/Users/jason/code/gomoku/scripts/autolab_monitor.py</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>600</integer>
  <key>WorkingDirectory</key><string>/Users/jason/code/gomoku</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>AUTOLAB_HOME</key><string>/Users/jason/data/autolab</string>
    <key>HOME</key><string>/Users/jason</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>StandardOutPath</key><string>/Users/jason/data/autolab/monitor/launchd.out.log</string>
  <key>StandardErrorPath</key><string>/Users/jason/data/autolab/monitor/launchd.err.log</string>
</dict>
</plist>
```

`com.gomoku.autolab.research.plist` — identical to monitor's shape except
`Label` = `com.gomoku.autolab.research`, `ProgramArguments` =
`[/opt/homebrew/bin/uv, run, python, -m, gomoku.lab.research, --once]`, `StartInterval` = `1800`,
and `Standard{Out,Err}Path` → `research/launchd.{out,err}.log`. The monitor +
research agents touch no GPU and need no MPS env (pure read + write + notify).

> **`HOME` in the monitor plist** is what lets `osascript` display notifications
> in the logged-in user's session.

