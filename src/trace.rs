use crossterm::style::{style, Color, Styler};
use std::fmt::{self, Write as _};
use std::io::{stdout, Stdout, Write};
use tracing::field::{Field, Visit};
use tracing::span::Attributes;
use tracing::{Event, Id, Subscriber};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

pub struct Tree {
    indent: u8,
    stdout: Stdout,
}

struct SpanData {
    format: String,
}

struct FieldVisitor {
    header: String,
    fields: String,
}

impl Tree {
    pub fn new(indent: u8) -> Tree {
        Tree {
            indent,
            stdout: stdout(),
        }
    }

    fn print_indent(&self, writer: &mut impl Write, amount: usize) {
        write!(
            writer,
            "{:indent$}",
            "",
            indent = self.indent as usize * amount
        )
        .unwrap();
    }
}

impl SpanData {
    pub fn new(attrs: &Attributes) -> SpanData {
        let mut visitor = FieldVisitor::new();
        attrs.record(&mut visitor);
        SpanData {
            format: visitor.finish(attrs.metadata().name(), Color::Green),
        }
    }
}

impl FieldVisitor {
    pub fn new() -> FieldVisitor {
        FieldVisitor {
            header: String::new(),
            fields: String::new(),
        }
    }

    pub fn finish(self, name: &str, color: Color) -> String {
        let header = if self.header.is_empty() {
            name
        } else {
            &self.header
        };
        format!(
            "{} {}{}{}",
            style(header).with(color),
            '['.bold(),
            self.fields,
            ']'.bold()
        )
    }
}

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        match field.name() {
            "message" => self.header = format!("{:?}", value),
            _ => {
                if !self.fields.is_empty() {
                    self.fields.push_str(", ");
                }
                write!(&mut self.fields, "{}={:?}", field.name().bold(), (value)).unwrap_or(());
            }
        }
    }
}

impl<S> Layer<S> for Tree
where
    S: Subscriber,
    S: for<'a> LookupSpan<'a>,
{
    fn new_span(&self, attrs: &Attributes, id: &Id, ctx: Context<S>) {
        let span = ctx.span(id).expect("created span, but didn't exist");
        let data = SpanData::new(attrs);
        span.extensions_mut().insert(data);
    }

    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).expect("span does not exist");
        let ext = span.extensions();
        let data = ext.get::<SpanData>().expect("span did not have data");

        let indentation = ctx.scope().count().saturating_sub(1);
        let mut stdout = self.stdout.lock();
        self.print_indent(&mut stdout, indentation);
        writeln!(&mut stdout, "{}", data.format).unwrap();
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::new();
        event.record(&mut visitor);
        let format = visitor.finish(event.metadata().name(), Color::Blue);

        let indentation = ctx.scope().count();
        let mut stdout = self.stdout.lock();
        self.print_indent(&mut stdout, indentation);
        writeln!(&mut stdout, "{}", format).unwrap();
    }
}
