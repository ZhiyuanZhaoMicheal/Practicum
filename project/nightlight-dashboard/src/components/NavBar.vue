<template>
  <nav class="navbar">
    <!-- Logo -->
    <RouterLink to="/" class="navbar__logo">
      <span class="navbar__logo-icon">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
          <circle cx="9" cy="9" r="8" stroke="currentColor" stroke-width="1.2" />
          <circle cx="9" cy="9" r="3" fill="currentColor" />
          <line x1="9" y1="1" x2="9" y2="4" stroke="currentColor" stroke-width="1.2" />
          <line x1="9" y1="14" x2="9" y2="17" stroke="currentColor" stroke-width="1.2" />
          <line x1="1" y1="9" x2="4" y2="9" stroke="currentColor" stroke-width="1.2" />
          <line x1="14" y1="9" x2="17" y2="9" stroke="currentColor" stroke-width="1.2" />
        </svg>
      </span>
      <span class="navbar__logo-text">NIGHTLIGHT</span>
    </RouterLink>

    <!-- Navigation links (desktop) -->
    <div class="navbar__links">
      <RouterLink
        v-for="link in links"
        :key="link.to"
        :to="link.to"
        class="navbar__link"
        :class="{ active: route.path === link.to }"
      >
        <span class="navbar__link-dot" />
        {{ link.label }}
      </RouterLink>
    </div>

    <!-- Right: status indicator -->
    <div class="navbar__status">
      <span class="status-dot" />
      <span class="status-text mono">{{ eventCount }} EVENTS LOADED</span>
    </div>

    <!-- Hamburger button (mobile) -->
    <button class="navbar__hamburger" @click="mobileOpen = !mobileOpen" aria-label="Menu">
      <span :class="{ open: mobileOpen }" />
    </button>

    <!-- Mobile dropdown -->
    <div class="navbar__mobile" :class="{ open: mobileOpen }">
      <RouterLink
        v-for="link in links"
        :key="'m-' + link.to"
        :to="link.to"
        class="navbar__mobile-link"
        :class="{ active: route.path === link.to }"
        @click="mobileOpen = false"
      >
        {{ link.label }}
      </RouterLink>
    </div>
  </nav>
</template>

<script setup>
import { ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { EVENTS } from '@/data/events.js'

const route = useRoute()
const eventCount = EVENTS.length
const mobileOpen = ref(false)

watch(() => route.path, () => { mobileOpen.value = false })

const links = [
  { to: '/',       label: 'Overview' },
  { to: '/map',    label: 'Map'      },
  { to: '/charts', label: 'Charts'   },
  { to: '/docs',   label: 'Docs'     },
]
</script>

<style scoped>
.navbar {
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 100;
  height: var(--nav-h);
  display: flex;
  align-items: center;
  gap: 32px;
  padding: 0 24px;
  background: rgba(3, 13, 26, 0.92);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
}

/* Logo */
.navbar__logo {
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--cyan);
  text-decoration: none;
  flex-shrink: 0;
}
.navbar__logo-icon {
  display: flex;
  align-items: center;
  animation: pulse-glow 3s ease infinite;
}
.navbar__logo-text {
  font-family: var(--font-head);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.18em;
  color: var(--text-bright);
}

/* Links */
.navbar__links {
  display: flex;
  align-items: center;
  gap: 4px;
}
.navbar__link {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 14px;
  border-radius: var(--radius);
  font-family: var(--font-head);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-muted);
  text-decoration: none;
  transition: all var(--t-fast);
  border: 1px solid transparent;
}
.navbar__link:hover {
  color: var(--text-bright);
  background: var(--bg-3);
  border-color: var(--border);
}
.navbar__link.active {
  color: var(--cyan);
  background: var(--cyan-dim);
  border-color: rgba(0,212,255,.2);
}
.navbar__link-dot {
  width: 5px; height: 5px;
  border-radius: 50%;
  background: currentColor;
  opacity: 0;
  transition: opacity var(--t-fast);
}
.navbar__link.active .navbar__link-dot { opacity: 1; }

/* Status */
.navbar__status {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
}
.status-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 6px var(--green);
  animation: blink 2.5s ease infinite;
}
.status-text {
  font-size: 10px;
  letter-spacing: 0.12em;
  color: var(--text-muted);
}

/* Hamburger button — hidden on desktop */
.navbar__hamburger {
  display: none;
  background: none;
  border: none;
  cursor: pointer;
  width: 28px;
  height: 28px;
  position: relative;
  margin-left: auto;
}
.navbar__hamburger span,
.navbar__hamburger span::before,
.navbar__hamburger span::after {
  display: block;
  width: 20px;
  height: 2px;
  background: var(--text);
  border-radius: 1px;
  position: absolute;
  left: 4px;
  transition: all 0.25s ease;
}
.navbar__hamburger span { top: 13px; }
.navbar__hamburger span::before { content: ''; top: -6px; }
.navbar__hamburger span::after  { content: ''; top: 6px; }
.navbar__hamburger span.open { background: transparent; }
.navbar__hamburger span.open::before { top: 0; transform: rotate(45deg); }
.navbar__hamburger span.open::after  { top: 0; transform: rotate(-45deg); }

/* Mobile dropdown — hidden on desktop */
.navbar__mobile {
  display: none;
}

@media (max-width: 768px) {
  .navbar { padding: 0 16px; }
  .navbar__links  { display: none; }
  .navbar__status { display: none; }
  .navbar__hamburger { display: block; }

  .navbar__mobile {
    display: flex;
    flex-direction: column;
    position: absolute;
    top: var(--nav-h);
    left: 0; right: 0;
    background: rgba(3, 13, 26, 0.97);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 8px 16px;
    gap: 2px;
    transform: translateY(-100%);
    opacity: 0;
    pointer-events: none;
    transition: transform 0.25s ease, opacity 0.25s ease;
    z-index: 99;
  }
  .navbar__mobile.open {
    transform: translateY(0);
    opacity: 1;
    pointer-events: all;
  }
  .navbar__mobile-link {
    display: block;
    padding: 10px 14px;
    font-family: var(--font-head);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    text-decoration: none;
    border-radius: var(--radius);
    transition: all var(--t-fast);
  }
  .navbar__mobile-link:hover,
  .navbar__mobile-link.active {
    color: var(--cyan);
    background: var(--cyan-dim);
  }
}
</style>
