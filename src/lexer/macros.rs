
macro_rules! keywords {
    (
        $vis:vis enum $ident:ident {
            $(
                $keyword:ident = $text:literal,
            )*
            $(
                #[otherwise]
                $kind:ident ($ty:ty),
            )*
        }
    ) => {
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
        $vis enum $ident {
            $($keyword,)*
            $($kind($ty),)*
        }

        impl $ident {
            fn parse(input: &str) -> PResult<$ident> {
                alt((
                    $(
                        map(tag($text), |_| $ident::$keyword),
                    )*
                    $(
                        map(<$ty>::parse, $ident::$kind),
                    )*
                ))(input)
            }
        }

        impl Display for $ident {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                match self {
                    $(
                        $ident::$keyword => write!(f, "`{}`", $text),
                    )*
                    $(
                        $ident::$kind(inner) => inner.fmt(f),
                    )*
                }
            }
        }
    }
}

macro_rules! symbols {
    (
        $vis:vis enum $ident:ident {
            $(
                $symbol:ident = $pattern:expr,
            )*
        }
    ) => {
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
        pub enum Symbol {
            Char(char),

            $($symbol),*
        }

        impl Symbol {
            fn parse(input: &str) -> PResult<Symbol> {
                alt((
                    $(
                        map(tag($pattern), |_| Symbol::$symbol),
                    )*
                    map(anychar, Symbol::Char),
                ))(input)
            }
        }

        impl Display for Symbol {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                match self {
                    Symbol::Char(ch) => write!(f, "`{}`", ch),
                    $(
                        Symbol::$symbol => write!(f, "`{}`", $pattern),
                    )*
                }
            }
        }

    };
}

