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

    <!-- Navigation links -->
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
  </nav>
</template>

<script setup>
import { useRoute } from 'vue-router'
import { EVENTS } from '@/data/events.js'

const route = useRoute()
const eventCount = EVENTS.length

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
</style>
