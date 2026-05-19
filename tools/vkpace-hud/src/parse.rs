//! Hand-rolled parser for the vkpace telemetry line.
//!
//! The layer formats one line per present as (`src/telemetry.rs:383` in
//! the layer crate):
//!
//! ```text
//! {"ts":<u64>,"frame":<u64>,"queue":"0x<hex>","pid":<u64>,"latency_us":<u64>}
//! ```
//!
//! Schema is closed and tiny; no serde dependency.

use crate::state::Record;

/// Max accepted line length. The layer never emits anything close to this;
/// a longer line means either driver-injected garbage on the socket or a
/// future schema we don't speak — drop it.
const MAX_LINE_BYTES: usize = 1024;

pub fn parse_line(line: &str) -> Option<Record> {
    if line.len() > MAX_LINE_BYTES {
        return None;
    }
    // Strip the wrapping braces + any trailing CR/LF the line-reader left.
    let s = line.trim_end_matches(['\n', '\r']).trim();
    let s = s.strip_prefix('{')?.strip_suffix('}')?;

    let mut rec = Record::default();
    let mut saw_any = false;
    for part in s.split(',') {
        let (k, v) = part.split_once(':')?;
        let key = k.trim().trim_matches('"');
        let val = v.trim();
        saw_any = true;
        match key {
            "ts" => rec.ts_ns = parse_u64(val)?,
            "frame" => rec.frame = parse_u64(val)?,
            "queue" => rec.queue = parse_queue(val)?,
            "pid" => rec.pid = parse_u64(val)?,
            "latency_us" => rec.latency_us = parse_u64(val)?,
            // Unknown keys are accepted silently — forwards-compat with a
            // future layer that adds fields. Missing-keys default to 0
            // (covered by `Record::default()`).
            _ => {}
        }
    }
    if !saw_any {
        return None;
    }
    Some(rec)
}

fn parse_u64(s: &str) -> Option<u64> {
    s.trim().parse::<u64>().ok()
}

/// `queue` is `"0x<hex>"`. Accept both quoted and unquoted forms so a
/// future layer that emits a decimal value still parses.
fn parse_queue(s: &str) -> Option<u64> {
    let s = s.trim().trim_matches('"');
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u64>().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirror the exact format the layer emits at `telemetry.rs:383`. If
    /// this test starts failing, the layer's schema changed and the parser
    /// needs the same update.
    #[test]
    fn round_trip_layer_format() {
        let line = r#"{"ts":1234567890,"frame":7,"queue":"0xdeadbeef","pid":42,"latency_us":1500}"#;
        let r = parse_line(line).expect("must parse");
        assert_eq!(r.ts_ns, 1_234_567_890);
        assert_eq!(r.frame, 7);
        assert_eq!(r.queue, 0xdead_beef);
        assert_eq!(r.pid, 42);
        assert_eq!(r.latency_us, 1500);
    }

    #[test]
    fn trailing_newline_accepted() {
        let line = "{\"ts\":1,\"frame\":2,\"queue\":\"0x0\",\"pid\":3,\"latency_us\":4}\n";
        assert!(parse_line(line).is_some());
    }

    #[test]
    fn missing_field_defaults_zero() {
        // `latency_us` omitted — layer never does this today, but we
        // accept it so future-us can add/remove fields without breaking
        // older HUD binaries in the field.
        let line = r#"{"ts":1,"frame":2,"queue":"0x0","pid":3}"#;
        let r = parse_line(line).unwrap();
        assert_eq!(r.latency_us, 0);
    }

    #[test]
    fn unknown_field_ignored() {
        let line = r#"{"ts":1,"frame":2,"queue":"0x0","pid":3,"latency_us":4,"future":99}"#;
        assert!(parse_line(line).is_some());
    }

    #[test]
    fn garbage_hex_rejects() {
        let line = r#"{"ts":1,"frame":2,"queue":"0xZZ","pid":3,"latency_us":4}"#;
        assert!(parse_line(line).is_none());
    }

    #[test]
    fn oversized_line_rejected() {
        // Anything past 1 KiB — the layer never emits this.
        let pad = "x".repeat(2000);
        let line =
            format!(r#"{{"ts":1,"frame":2,"queue":"0x0","pid":3,"latency_us":4,"x":"{pad}"}}"#);
        assert!(parse_line(&line).is_none());
    }

    #[test]
    fn missing_braces_rejected() {
        assert!(parse_line(r#""ts":1,"frame":2"#).is_none());
    }

    #[test]
    fn empty_object_rejected() {
        // `{}` has no keys — we treat that as malformed rather than a
        // record full of zeroes, so the HUD doesn't display phantom frames.
        assert!(parse_line("{}").is_none());
    }

    #[test]
    fn key_reorder_accepted() {
        let line = r#"{"pid":3,"queue":"0xff","latency_us":4,"frame":2,"ts":1}"#;
        let r = parse_line(line).unwrap();
        assert_eq!(r.ts_ns, 1);
        assert_eq!(r.queue, 0xff);
    }

    #[test]
    fn whitespace_tolerated() {
        let line = r#"{ "ts" : 1 , "frame" : 2 , "queue" : "0x0" , "pid" : 3 , "latency_us" : 4 }"#;
        assert!(parse_line(line).is_some());
    }
}
